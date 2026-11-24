// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::num::NonZeroU32;
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;

use arrow_array::{RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field as ArrowField};
use datafusion::common::Column;
use datafusion::error::DataFusionError;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_expr::Expr;
use futures::{Stream, StreamExt};
use lance_arrow::RecordBatchExt;
use lance_arrow::json::convert_lance_json_to_arrow;
use lance_core::ROW_ADDR;
use lance_core::datatypes::Schema;
use lance_datafusion::projection::{OutputColumn, ProjectionPlan};
use lance_table_runtime::{Table, TakeOptions, TakeProjection, TakeSelection};

use super::{
    Dataset, ProjectionRequest, fragment::FileFragment, scanner::DatasetRecordBatchStream,
};
use crate::{Error, Result};

pub async fn row_offsets_to_row_addresses(
    fragments: &[FileFragment],
    row_indices: &[u64],
) -> Result<Vec<u64>> {
    lance_table_runtime::take::row_offsets_to_row_addresses(
        fragments.iter().map(FileFragment::as_runtime),
        row_indices,
    )
    .await
}

pub async fn take(
    dataset: &Dataset,
    offsets: &[u64],
    projection: ProjectionRequest,
) -> Result<RecordBatch> {
    take_with_io_parallelism(dataset, offsets, projection, None).await
}

pub async fn take_with_io_parallelism(
    dataset: &Dataset,
    offsets: &[u64],
    projection: ProjectionRequest,
    io_parallelism: Option<u32>,
) -> Result<RecordBatch> {
    let io_parallelism = io_parallelism
        .map(|limit| {
            NonZeroU32::new(limit).ok_or_else(|| {
                Error::invalid_input("take io_parallelism must be greater than 0, got 0")
            })
        })
        .transpose()?;
    let table = dataset.table_view().to_table();
    match projection {
        ProjectionRequest::Schema(schema) => {
            table
                .take_with_io_parallelism(offsets, &schema, io_parallelism)
                .await
        }
        projection @ ProjectionRequest::Sql(_) => {
            let projection = projection.into_projection_plan(Arc::new(dataset.clone()))?;
            take_expressions(
                &table,
                TakeSelection::Indices(offsets),
                Arc::new(projection),
                TakeOptions {
                    io_parallelism,
                    ..Default::default()
                },
            )
            .await
        }
    }
}

async fn take_expressions(
    table: &Table,
    selection: TakeSelection<'_>,
    mut projection: Arc<ProjectionPlan>,
    options: TakeOptions,
) -> Result<RecordBatch> {
    let batch = table
        .take_physical(
            selection,
            &projection.physical_projection,
            projection.must_add_row_offset,
            options,
        )
        .await?;
    let Some(batch) = batch else {
        let mut batch = RecordBatch::new_empty(Arc::new(projection.output_schema()?));
        if options.with_row_address && batch.column_by_name(ROW_ADDR).is_none() {
            batch = batch.try_with_column(
                ArrowField::new(ROW_ADDR, DataType::UInt64, false),
                Arc::new(UInt64Array::from(Vec::<u64>::new())),
            )?;
        }
        return Ok(convert_lance_json_to_arrow(&batch)?);
    };

    if options.with_row_address
        && !projection
            .requested_output_expr
            .iter()
            .any(|c| c.name == ROW_ADDR)
    {
        Arc::make_mut(&mut projection)
            .requested_output_expr
            .push(OutputColumn {
                expr: Expr::Column(Column::from_name(ROW_ADDR)),
                name: ROW_ADDR.to_string(),
            });
    }
    Ok(convert_lance_json_to_arrow(
        &projection.project_batch(batch).await?,
    )?)
}

/// Get a stream of batches based on iterator of ranges of row numbers.
///
/// `batch_readahead` must be greater than zero; otherwise the stream yields an invalid-input error.
///
/// This is an experimental API. It may change at any time.
pub fn take_scan(
    dataset: &Dataset,
    row_ranges: Pin<Box<dyn Stream<Item = Result<Range<u64>>> + Send>>,
    projection: Arc<Schema>,
    batch_readahead: usize,
) -> DatasetRecordBatchStream {
    let arrow_schema = Arc::new(projection.as_ref().into());
    if batch_readahead == 0 {
        let error = DataFusionError::from(Error::invalid_input(
            "batch_readahead must be greater than 0, got 0",
        ));
        return DatasetRecordBatchStream::new(Box::pin(RecordBatchStreamAdapter::new(
            arrow_schema,
            futures::stream::once(std::future::ready(Err(error))),
        )));
    }
    let dataset = Arc::new(dataset.clone());
    let batch_stream = row_ranges
        .map(move |res| {
            let dataset = dataset.clone();
            let projection = projection.clone();
            let fut = async move {
                let range = res.map_err(|err| DataFusionError::External(Box::new(err)))?;
                let row_pos: Vec<u64> = (range.start..range.end).collect();
                dataset
                    .take(&row_pos, ProjectionRequest::Schema(projection.clone()))
                    .await
                    .map_err(|err| DataFusionError::External(Box::new(err)))
            };
            async move { lance_core::utils::tokio::spawn(fut).await.unwrap() }
        })
        .buffered(batch_readahead);

    DatasetRecordBatchStream::new(Box::pin(RecordBatchStreamAdapter::new(
        arrow_schema,
        batch_stream,
    )))
}

#[derive(Clone, Debug)]
enum TakeRows {
    Ids(Vec<u64>),
    Addresses(Vec<u64>),
}

#[derive(Clone, Debug)]
enum TakeOutput {
    Schema(TakeProjection),
    Expressions(Arc<ProjectionPlan>),
}

/// Builder for the `take` operation.
#[derive(Clone, Debug)]
pub struct TakeBuilder {
    dataset: Arc<Dataset>,
    rows: TakeRows,
    output: TakeOutput,
    with_row_address: bool,
    ignore_deletions: bool,
    io_parallelism: Option<u32>,
}

impl TakeBuilder {
    /// Create a new `TakeBuilder` for taking by id
    pub fn try_new_from_ids(
        dataset: Arc<Dataset>,
        row_ids: Vec<u64>,
        projection: ProjectionRequest,
    ) -> Result<Self> {
        let output = match projection {
            ProjectionRequest::Schema(schema) => {
                TakeOutput::Schema(TakeProjection::try_new(dataset.clone(), &schema)?)
            }
            projection @ ProjectionRequest::Sql(_) => {
                TakeOutput::Expressions(Arc::new(projection.into_projection_plan(dataset.clone())?))
            }
        };
        Ok(Self {
            dataset,
            rows: TakeRows::Ids(row_ids),
            output,
            with_row_address: false,
            ignore_deletions: false,
            io_parallelism: None,
        })
    }

    /// Create a new `TakeBuilder` for taking by address
    pub fn try_new_from_addresses(
        dataset: Arc<Dataset>,
        addresses: Vec<u64>,
        projection: Arc<ProjectionPlan>,
    ) -> Result<Self> {
        Ok(Self {
            dataset,
            rows: TakeRows::Addresses(addresses),
            output: TakeOutput::Expressions(projection),
            with_row_address: false,
            ignore_deletions: false,
            io_parallelism: None,
        })
    }

    /// Adds row addresses to the output
    pub fn with_row_address(mut self, with_row_address: bool) -> Self {
        self.with_row_address = with_row_address;
        self
    }

    /// Takes the rows as if the dataset had no deletions.
    ///
    /// Deletion files are read only where a row id could name more than one
    /// physical copy (fragments whose id spans overlap, as after an update),
    /// neither to filter the rows nor for the other fragments, so on a table
    /// with many fragments the take saves one request per fragment. A deleted
    /// row may come back with its stored values or be dropped, depending on
    /// whether its fragment's deletion vector was needed, so the output is
    /// only complete for live ids: use this only when the ids are known to be
    /// live, for example row ids from an index built after the last deletion.
    pub fn with_ignore_deletions(mut self, ignore_deletions: bool) -> Self {
        self.ignore_deletions = ignore_deletions;
        self
    }

    /// Sets the maximum active fragment-reader data reads in the primary base for this take.
    ///
    /// Current-format primary-base data and overlay files opened through the
    /// fragment reader share one scheduler with this request ceiling, and
    /// fragment-reader fanout is capped at `min(store_parallelism, io_parallelism)`.
    /// This does not increase fragment-reader fanout above the store default, so
    /// actual concurrency may be lower. Legacy v1 files, additional-base
    /// schedulers, and auxiliary metadata or lineage reads (such as deletion
    /// vectors and spilled row-ID or row-version sequences) keep their existing
    /// schedulers. Must be greater than zero.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use lance::{Dataset, Result};
    /// # async fn example(dataset: Arc<Dataset>) -> Result<()> {
    /// let batch = dataset
    ///     .take_builder(&[0, 1, 2], dataset.schema().clone())?
    ///     .with_io_parallelism(64)
    ///     .execute()
    ///     .await?;
    /// # let _ = batch;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_io_parallelism(mut self, io_parallelism: u32) -> Self {
        self.io_parallelism = Some(io_parallelism);
        self
    }

    /// Execute the take operation and return a single batch
    pub async fn execute(self) -> Result<RecordBatch> {
        let io_parallelism = self
            .io_parallelism
            .map(|limit| {
                NonZeroU32::new(limit).ok_or_else(|| {
                    Error::invalid_input("TakeBuilder io_parallelism must be greater than 0, got 0")
                })
            })
            .transpose()?;
        let options = TakeOptions {
            with_row_address: self.with_row_address,
            ignore_deletions: self.ignore_deletions,
            io_parallelism,
            ..Default::default()
        };
        let selection = match &self.rows {
            TakeRows::Ids(ids) => TakeSelection::RowIds(ids),
            TakeRows::Addresses(addrs) => TakeSelection::RowAddresses(addrs),
        };
        let table = self.dataset.table_view().to_table();
        match self.output {
            TakeOutput::Schema(projection) => {
                table
                    .take_with_options(selection, &projection, options)
                    .await
            }
            TakeOutput::Expressions(projection) => {
                take_expressions(&table, selection, projection, options).await
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        match &self.rows {
            TakeRows::Ids(rows) | TakeRows::Addresses(rows) => rows.is_empty(),
        }
    }
}

#[cfg(test)]
mod test {
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int64Type;
    use arrow_array::{Array, ArrayRef};
    use arrow_array::{
        Int32Array, LargeBinaryArray, ListArray, MapArray, RecordBatchIterator, StringArray,
        StructArray,
    };
    use arrow_buffer::OffsetBuffer;
    use arrow_schema::{DataType, Fields, Schema as ArrowSchema};
    use futures::TryStreamExt;
    use lance_arrow::ARROW_EXT_NAME_KEY;
    use lance_arrow::json::{ARROW_JSON_EXT_NAME, is_arrow_json_field, is_json_field};
    use lance_core::utils::address::RowAddress;
    use lance_core::utils::testing::{ProxyObjectStore, ProxyObjectStorePolicy};
    use lance_core::{ROW_ADDR_FIELD, ROW_ID_FIELD};
    use lance_file::version::LanceFileVersion;
    use lance_io::stream::RecordBatchStream;
    use object_store::throttle::{ThrottleConfig, ThrottledStore};
    use pretty_assertions::assert_eq;
    use rstest::rstest;
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use crate::dataset::{
        MergeInsertBuilder, UpdateBuilder, WhenMatched, WhenNotMatched, WriteParams,
        scanner::test_dataset::TestVectorDataset,
    };

    use super::*;

    #[rstest]
    #[case::empty(&[])]
    #[case::contiguous(&[0, 1])]
    #[case::sorted(&[0, 4, 4])]
    #[case::unsorted(&[4, 0, 4])]
    #[case::mixed_deleted(&[4, 1, 0, 4])]
    #[case::deleted_only(&[1, 1])]
    #[tokio::test]
    async fn test_take_empty_projection(
        #[case] indices: &[u64],
        #[values(false, true)] enable_stable_row_ids: bool,
        #[values(false, true)] delete_row: bool,
        #[values(false, true)] by_id: bool,
    ) {
        let data = test_batch(0..6);
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(data.clone())], data.schema()),
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 3,
                enable_stable_row_ids,
                data_storage_version: Some(LanceFileVersion::Stable),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.get_fragments().len(), 2);
        if delete_row {
            dataset.delete("i = 1").await.unwrap();
        }
        let expected_rows = if by_id && delete_row {
            indices.iter().filter(|&&id| id != 1).count()
        } else {
            indices.len()
        };
        let row_ids = indices
            .iter()
            .map(|&id| {
                if enable_stable_row_ids {
                    id
                } else {
                    u64::from(RowAddress::new_from_parts((id / 3) as u32, (id % 3) as u32))
                }
            })
            .collect::<Vec<_>>();
        let projection = ProjectionRequest::from_columns(Vec::<&str>::new(), dataset.schema());
        dataset.table_state.object_store.io_stats_incremental();
        let batch = if by_id {
            dataset.take_rows(&row_ids, projection).await.unwrap()
        } else {
            dataset.take(indices, projection).await.unwrap()
        };
        assert_eq!(batch.num_columns(), 0);
        assert_eq!(batch.num_rows(), expected_rows);
        assert!(
            dataset
                .table_state
                .object_store
                .io_stats_incremental()
                .requests
                .iter()
                .all(|request| !request.path.as_ref().contains("/data/"))
        );

        let projection = ProjectionRequest::from_sql([("answer", "42")]);
        let batch = if by_id {
            dataset.take_rows(&row_ids, projection).await.unwrap()
        } else {
            dataset.take(indices, projection).await.unwrap()
        };
        assert_eq!(batch.num_columns(), 1);
        assert_eq!(batch.num_rows(), expected_rows);
        assert_eq!(batch.schema().field(0).name(), "answer");
        assert_eq!(
            batch["answer"]
                .as_primitive::<Int64Type>()
                .values()
                .as_ref(),
            vec![42; expected_rows]
        );
        assert!(
            dataset
                .table_state
                .object_store
                .io_stats_incremental()
                .requests
                .iter()
                .all(|request| !request.path.as_ref().contains("/data/"))
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_take_empty_projection_bounds(#[values(false, true)] enable_stable_row_ids: bool) {
        let data = test_batch(0..6);
        let dataset = Dataset::write(
            RecordBatchIterator::new([Ok(data.clone())], data.schema()),
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 3,
                enable_stable_row_ids,
                data_storage_version: Some(LanceFileVersion::Stable),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.get_fragments().len(), 2);
        let projection = ProjectionRequest::from_columns(Vec::<&str>::new(), dataset.schema());
        let error = dataset.take(&[6], projection.clone()).await.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error.to_string().contains("non-existent fragment"),
            "{error}"
        );

        if enable_stable_row_ids {
            let batch = dataset.take_rows(&[6], projection.clone()).await.unwrap();
            assert_eq!((batch.num_rows(), batch.num_columns()), (0, 0));
            let batch = dataset.take_rows(&[6, 0, 0], projection).await.unwrap();
            assert_eq!((batch.num_rows(), batch.num_columns()), (2, 0));
        } else {
            let error = dataset
                .take_rows(&[3], projection.clone())
                .await
                .unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
            assert!(error.to_string().contains("Invalid read params"), "{error}");
            let missing = u64::from(RowAddress::new_from_parts(2, 0));
            let error = dataset.take_rows(&[missing], projection).await.unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
            assert!(
                error.to_string().contains("non-existent fragment: 2"),
                "{error}"
            );
        }
    }

    #[rstest]
    #[case::single(&[6])]
    #[case::repeated(&[6, 6])]
    #[case::sorted(&[0, 6])]
    #[case::unsorted(&[6, 0])]
    #[tokio::test]
    async fn test_take_out_of_bound(
        #[case] indices: &[u64],
        #[values(false, true)] enable_stable_row_ids: bool,
    ) {
        let data = test_batch(0..6);
        let dataset = Dataset::write(
            RecordBatchIterator::new([Ok(data.clone())], data.schema()),
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 3,
                enable_stable_row_ids,
                data_storage_version: Some(LanceFileVersion::Stable),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.get_fragments().len(), 2);

        let error = dataset
            .take(indices, dataset.schema().clone())
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains("fragment"), "{error}");
        assert!(error.to_string().contains("4294967295"), "{error}");
    }

    #[rstest]
    #[case::empty(vec![])]
    #[case::missing(vec![100])]
    #[tokio::test]
    async fn test_take_empty_row_address_schema(
        #[case] ids: Vec<u64>,
        #[values(false, true)] explicit_address: bool,
        #[values(false, true)] with_row_address: bool,
    ) {
        let data = test_batch(0..4);
        let dataset = Arc::new(
            Dataset::write(
                RecordBatchIterator::new([Ok(data.clone())], data.schema()),
                "memory://",
                Some(WriteParams {
                    max_rows_per_file: 2,
                    enable_stable_row_ids: true,
                    data_storage_version: Some(LanceFileVersion::Stable),
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );
        assert_eq!(dataset.get_fragments().len(), 2);
        let columns = if explicit_address {
            vec!["i", ROW_ADDR]
        } else {
            vec!["i"]
        };
        let projection = ProjectionRequest::from_columns(columns, dataset.schema());
        let expected = dataset
            .take_builder(&[0], projection.clone())
            .unwrap()
            .with_row_address(with_row_address)
            .execute()
            .await
            .unwrap();
        let actual = dataset
            .take_builder(&ids, projection)
            .unwrap()
            .with_row_address(with_row_address)
            .execute()
            .await
            .unwrap();
        assert_eq!(actual.num_rows(), 0);
        assert_eq!(actual.schema(), expected.schema());
    }

    // Used to validate that futures returned are Send.
    fn require_send<T: Send>(t: T) -> T {
        t
    }

    fn test_batch(i_range: Range<i32>) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("i", DataType::Int32, false),
            ArrowField::new("s", DataType::Utf8, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from_iter_values(i_range.clone())),
                Arc::new(StringArray::from_iter_values(
                    i_range.map(|i| format!("str-{}", i)),
                )),
            ],
        )
        .unwrap()
    }

    #[rstest]
    #[case::top_level(&["i", "nope"], Some("nope"))]
    #[case::nested(&["st.a", "st.nope"], Some("st.nope"))]
    #[case::under_a_leaf(&["i.a"], Some("i.a"))]
    #[case::nested_and_system(&["st.a", "_rowid"], None)]
    #[tokio::test]
    async fn test_take_projection_from_column_names(
        #[case] columns: &[&str],
        #[case] missing: Option<&str>,
    ) {
        let child = Arc::new(ArrowField::new("a", DataType::Int32, false));
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("i", DataType::Int32, false),
            ArrowField::new("st", DataType::Struct(vec![child.clone()].into()), false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![0, 1])),
                Arc::new(StructArray::new(
                    vec![child].into(),
                    vec![Arc::new(Int32Array::from(vec![10, 11]))],
                    None,
                )),
            ],
        )
        .unwrap();
        let reader = RecordBatchIterator::new([Ok(batch)], schema);
        let dataset = Dataset::write(reader, "memory://", None).await.unwrap();

        let projection = ProjectionRequest::try_from_columns(columns, dataset.schema());
        match missing {
            Some(missing) => {
                let err = projection.unwrap_err();
                assert!(matches!(err, Error::FieldNotFound { .. }), "{err:?}");
                assert!(err.to_string().contains(missing), "{err}");
            }
            None => {
                let values = dataset.take(&[1], projection.unwrap()).await.unwrap();
                let st = values["st"].as_struct();
                assert_eq!(st.column_names(), vec!["a"]);
                assert_eq!(
                    st.column(0)
                        .as_primitive::<arrow_array::types::Int32Type>()
                        .value(0),
                    11
                );
                assert!(values.column_by_name("_rowid").is_some());
            }
        }
    }

    /// How a nested field carries one Arrow JSON document per row.
    #[derive(Clone, Copy, Debug)]
    enum NestedJson {
        /// `struct<doc: json>`
        Struct,
        /// `list<struct<doc: json>>` with one element per row
        ListOfStruct,
        /// `map<string, json>` with one entry per row
        Map,
    }

    fn arrow_json_field(name: &str) -> ArrowField {
        ArrowField::new(name, DataType::Utf8, true).with_metadata(HashMap::from([(
            ARROW_EXT_NAME_KEY.to_string(),
            ARROW_JSON_EXT_NAME.to_string(),
        )]))
    }

    fn nested_json_array(shape: NestedJson, docs: &[String]) -> ArrayRef {
        let docs: ArrayRef = Arc::new(StringArray::from_iter_values(docs));
        let one_per_row = || OffsetBuffer::from_lengths(std::iter::repeat_n(1, docs.len()));
        let doc_struct = || {
            StructArray::new(
                Fields::from(vec![arrow_json_field("doc")]),
                vec![docs.clone()],
                None,
            )
        };
        match shape {
            NestedJson::Struct => Arc::new(doc_struct()),
            NestedJson::ListOfStruct => {
                let values = doc_struct();
                let item = Arc::new(ArrowField::new("item", values.data_type().clone(), true));
                Arc::new(ListArray::new(item, one_per_row(), Arc::new(values), None))
            }
            NestedJson::Map => {
                let keys: ArrayRef = Arc::new(StringArray::from_iter_values(std::iter::repeat_n(
                    "k",
                    docs.len(),
                )));
                let entry_fields = Fields::from(vec![
                    ArrowField::new("key", DataType::Utf8, false),
                    arrow_json_field("value"),
                ]);
                let entries = StructArray::new(entry_fields, vec![keys, docs.clone()], None);
                let entries_field = Arc::new(ArrowField::new(
                    "entries",
                    entries.data_type().clone(),
                    false,
                ));
                Arc::new(MapArray::new(
                    entries_field,
                    one_per_row(),
                    entries,
                    None,
                    false,
                ))
            }
        }
    }

    /// The JSON leaf of `payload`, which a read must return as Arrow JSON.
    fn nested_json_leaf(shape: NestedJson, payload: &ArrowField) -> ArrowField {
        let child = |data_type: &DataType| match data_type {
            DataType::Struct(fields) => fields.clone(),
            other => panic!("expected struct, got {other}"),
        };
        match (shape, payload.data_type()) {
            (NestedJson::Struct, DataType::Struct(fields)) => fields[0].as_ref().clone(),
            (NestedJson::ListOfStruct, DataType::List(item)) => {
                child(item.data_type())[0].as_ref().clone()
            }
            (NestedJson::Map, DataType::Map(entries, _)) => {
                child(entries.data_type())[1].as_ref().clone()
            }
            (shape, other) => panic!("unexpected {shape:?} payload type {other}"),
        }
    }

    fn nested_json_doc(shape: NestedJson, payload: &ArrayRef, row: usize) -> String {
        let docs = match shape {
            NestedJson::Struct => payload.as_struct().column(0).clone(),
            NestedJson::ListOfStruct => payload
                .as_list::<i32>()
                .value(row)
                .as_struct()
                .column(0)
                .clone(),
            NestedJson::Map => payload.as_map().value(row).column(1).clone(),
        };
        let row = if matches!(shape, NestedJson::Struct) {
            row
        } else {
            0
        };
        docs.as_string::<i32>().value(row).to_string()
    }

    /// Rows `(id, label, payload)` whose payload holds `{"v":<version>}`.
    fn nested_json_batch(shape: NestedJson, ids: &[i32], version: i32) -> RecordBatch {
        let docs = ids
            .iter()
            .map(|id| format!(r#"{{"v":{}}}"#, version + id))
            .collect::<Vec<_>>();
        let payload = nested_json_array(shape, &docs);
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, false),
            ArrowField::new("label", DataType::Utf8, true),
            ArrowField::new("payload", payload.data_type().clone(), true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(ids.to_vec())),
                Arc::new(StringArray::from_iter_values(
                    ids.iter().map(|_| "original"),
                )),
                payload,
            ],
        )
        .unwrap()
    }

    #[rstest]
    #[tokio::test]
    async fn test_take(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
        #[values(false, true)] enable_stable_row_ids: bool,
    ) {
        let data = test_batch(0..400);
        let write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            data_storage_version: Some(data_storage_version),
            enable_stable_row_ids,
            ..Default::default()
        };
        let batches = RecordBatchIterator::new([Ok(data.clone())], data.schema());
        let dataset = Dataset::write(batches, "memory://", Some(write_params))
            .await
            .unwrap();

        assert_eq!(dataset.count_rows(None).await.unwrap(), 400);
        let projection = Schema::try_from(data.schema().as_ref()).unwrap();
        let values = dataset
            .take(
                &[
                    200, // 200
                    199, // 199
                    39,  // 39
                    40,  // 40
                    199, // 40
                    40,  // 40
                    125, // 125
                ],
                projection,
            )
            .await
            .unwrap();
        assert_eq!(
            RecordBatch::try_new(
                data.schema(),
                vec![
                    Arc::new(Int32Array::from_iter_values([
                        200, 199, 39, 40, 199, 40, 125
                    ])),
                    Arc::new(StringArray::from_iter_values(
                        [200, 199, 39, 40, 199, 40, 125]
                            .iter()
                            .map(|v| format!("str-{v}"))
                    )),
                ],
            )
            .unwrap(),
            values
        );
    }

    /// Nested Arrow JSON must be stored as JSONB by every write path and read
    /// back as Arrow JSON. Maps need file format 2.2.
    #[rstest]
    #[case::struct_json(NestedJson::Struct)]
    #[case::list_struct_json(NestedJson::ListOfStruct)]
    #[case::map_string_json(NestedJson::Map)]
    #[tokio::test]
    async fn test_nested_arrow_json_write_paths(
        #[case] shape: NestedJson,
        #[values(LanceFileVersion::V2_2, LanceFileVersion::V2_3)]
        data_storage_version: LanceFileVersion,
    ) {
        let write_params = WriteParams {
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        let created = nested_json_batch(shape, &[0, 1], 0);
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(created.clone())], created.schema()),
            "memory://",
            Some(write_params.clone()),
        )
        .await
        .unwrap();
        let appended = nested_json_batch(shape, &[2, 3], 0);
        dataset
            .append(
                RecordBatchIterator::new([Ok(appended.clone())], appended.schema()),
                Some(write_params),
            )
            .await
            .unwrap();
        let stored_schema = ArrowSchema::from(dataset.schema());
        assert!(is_json_field(&nested_json_leaf(
            shape,
            stored_schema.field_with_name("payload").unwrap()
        )));

        // Rewrites the row with id 1, JSON payload included, into a new fragment.
        let dataset = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id = 1")
            .unwrap()
            .set("label", "'updated'")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap()
            .new_dataset;

        // A sub-schema source rewrites the payload column in place: every row of
        // the appended fragment, and one row of the created fragment.
        let source = nested_json_batch(shape, &[0, 2, 3], 100)
            .project(&[0, 2])
            .unwrap();
        let mut merge = MergeInsertBuilder::try_new(dataset, vec!["id".to_string()]).unwrap();
        merge
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::DoNothing);
        let (dataset, stats) = merge
            .try_build()
            .unwrap()
            .execute_reader(RecordBatchIterator::new(
                [Ok(source.clone())],
                source.schema(),
            ))
            .await
            .unwrap();
        assert_eq!(stats.num_updated_rows, 3);

        let projection = Schema::try_from(created.schema().as_ref()).unwrap();
        let values = dataset
            .take(&[0, 1, 2, 3], projection.clone())
            .await
            .unwrap();
        let values_schema = values.schema();
        let payload_field = values_schema.field_with_name("payload").unwrap();
        assert!(is_arrow_json_field(&nested_json_leaf(shape, payload_field)));
        let ids = values["id"].as_primitive::<arrow_array::types::Int32Type>();
        let labels = values["label"].as_string::<i32>();
        let mut rows = (0..values.num_rows())
            .map(|row| {
                (
                    ids.value(row),
                    labels.value(row).to_string(),
                    nested_json_doc(shape, &values["payload"], row),
                )
            })
            .collect::<Vec<_>>();
        rows.sort();
        let expected = [
            (0, "original", r#"{"v":100}"#),
            (1, "updated", r#"{"v":1}"#),
            (2, "original", r#"{"v":102}"#),
            (3, "original", r#"{"v":103}"#),
        ]
        .map(|(id, label, doc)| (id, label.to_string(), doc.to_string()));
        assert_eq!(rows, expected);

        let empty = dataset.take(&[], projection).await.unwrap();
        assert_eq!(empty.num_rows(), 0);
        assert_eq!(empty.schema(), values_schema);
    }

    #[tokio::test]
    async fn test_take_with_deletion() {
        let data = test_batch(0..120);
        let write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            ..Default::default()
        };
        let batches = RecordBatchIterator::new([Ok(data.clone())], data.schema());
        let mut dataset = Dataset::write(batches, "memory://", Some(write_params))
            .await
            .unwrap();

        dataset.delete("i in (40, 77, 78, 79)").await.unwrap();

        let projection = Schema::try_from(data.schema().as_ref()).unwrap();
        let values = dataset
            .take(
                &[
                    0,   // 0
                    39,  // 39
                    40,  // 41
                    75,  // 76
                    76,  // 80
                    77,  // 81
                    115, // 119
                ],
                projection,
            )
            .await
            .unwrap();

        assert_eq!(
            RecordBatch::try_new(
                data.schema(),
                vec![
                    Arc::new(Int32Array::from_iter_values([0, 39, 41, 76, 80, 81, 119])),
                    Arc::new(StringArray::from_iter_values(
                        [0, 39, 41, 76, 80, 81, 119]
                            .iter()
                            .map(|v| format!("str-{v}"))
                    )),
                ],
            )
            .unwrap(),
            values
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_take_with_projection(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
        #[values(false, true)] enable_stable_row_ids: bool,
    ) {
        let data = test_batch(0..400);
        let write_params = WriteParams {
            data_storage_version: Some(data_storage_version),
            enable_stable_row_ids,
            ..Default::default()
        };
        let batches = RecordBatchIterator::new([Ok(data.clone())], data.schema());
        let dataset = Dataset::write(batches, "memory://", Some(write_params))
            .await
            .unwrap();

        assert_eq!(dataset.count_rows(None).await.unwrap(), 400);
        let projection =
            ProjectionRequest::from_sql([("foo", "i"), ("bar", "i*2"), ("offset", "_rowoffset")]);
        let values = dataset
            .take(&[10, 50, 100], projection.clone())
            .await
            .unwrap();

        let expected_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("foo", DataType::Int32, false),
            ArrowField::new("bar", DataType::Int32, false),
            ArrowField::new("offset", DataType::UInt64, false),
        ]));
        assert_eq!(
            RecordBatch::try_new(
                expected_schema,
                vec![
                    Arc::new(Int32Array::from_iter_values([10, 50, 100])),
                    Arc::new(Int32Array::from_iter_values([20, 100, 200])),
                    Arc::new(UInt64Array::from(vec![10, 50, 100])),
                ],
            )
            .unwrap(),
            values
        );

        let values2 = dataset.take_rows(&[10, 50, 100], projection).await.unwrap();
        assert_eq!(values, values2);
    }

    #[tokio::test]
    async fn test_take_legacy_blob_input_on_v2_2() {
        let mut metadata = HashMap::new();
        metadata.insert(lance_arrow::BLOB_META_KEY.to_string(), "true".to_string());

        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("blob", DataType::LargeBinary, true).with_metadata(metadata),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(LargeBinaryArray::from(vec![Some(
                b"hello".as_slice(),
            )]))],
        )
        .unwrap();

        let write_params = WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        };
        let batches = RecordBatchIterator::new([Ok(batch)], schema);
        let dataset = Arc::new(
            Dataset::write(batches, "memory://", Some(write_params))
                .await
                .unwrap(),
        );
        let blobs = dataset.take_blobs_by_indices(&[0], "blob").await.unwrap();
        assert_eq!(
            blobs[0].as_ref().unwrap().read().await.unwrap().as_ref(),
            b"hello"
        );
    }

    #[tokio::test]
    async fn test_take_blob_v2_from_blob_v2_struct_on_v2_2() {
        let schema = Arc::new(ArrowSchema::new(vec![crate::blob::blob_field(
            "blob", true,
        )]));
        let mut builder = crate::blob::BlobArrayBuilder::new(1);
        builder.push_bytes(b"hello").unwrap();
        let array = builder.finish().unwrap();

        let batch = RecordBatch::try_new(schema.clone(), vec![array]).unwrap();
        let write_params = WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        };
        let batches = RecordBatchIterator::new([Ok(batch)], schema);
        let dataset = crate::dataset::write::InsertBuilder::new("memory://")
            .with_params(&write_params)
            .execute_stream(batches)
            .await
            .unwrap();

        let proj = ProjectionRequest::from_columns(["blob"], dataset.schema());
        let values = dataset.take(&[0u64], proj).await.unwrap();

        let struct_arr = values.column(0).as_struct();
        assert_eq!(struct_arr.fields().len(), 5);
        assert_eq!(struct_arr.fields()[0].name(), "kind");
        assert_eq!(struct_arr.fields()[1].name(), "position");
        assert_eq!(struct_arr.fields()[2].name(), "size");
        assert_eq!(struct_arr.fields()[3].name(), "blob_id");
        assert_eq!(struct_arr.fields()[4].name(), "blob_uri");
    }

    #[tokio::test]
    async fn test_projection_plan_accepts_unloaded_legacy_blob_schema() {
        let mut metadata = HashMap::new();
        metadata.insert(lance_arrow::BLOB_META_KEY.to_string(), "true".to_string());
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("blob", DataType::LargeBinary, true).with_metadata(metadata),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(LargeBinaryArray::from(vec![Some(
                b"hello".as_slice(),
            )]))],
        )
        .unwrap();
        let write_params = WriteParams {
            data_storage_version: Some(LanceFileVersion::Legacy),
            ..Default::default()
        };
        let batches = RecordBatchIterator::new([Ok(batch)], schema);
        let dataset = Dataset::write(batches, "memory://", Some(write_params))
            .await
            .unwrap();

        let mut projection = dataset.schema().project(&["blob"]).unwrap();
        projection.fields[0].unloaded_mut();

        let projection = ProjectionRequest::from_schema(projection)
            .into_projection_plan(Arc::new(dataset))
            .unwrap();

        let output_schema = projection.output_schema().unwrap();
        let blob_field = output_schema.field_with_name("blob").unwrap();
        let DataType::Struct(fields) = blob_field.data_type() else {
            panic!("expected blob output schema to be a struct, got {blob_field:?}");
        };
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].name(), "position");
        assert_eq!(fields[1].name(), "size");
    }

    #[rstest]
    #[tokio::test]
    async fn test_take_rowid_rowaddr_with_projection_enable_stable_row_ids_projection_from_sql(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let data = test_batch(0..400);
        let write_params = WriteParams {
            data_storage_version: Some(data_storage_version),
            enable_stable_row_ids: true,
            max_rows_per_file: 50,
            ..Default::default()
        };
        let batches = RecordBatchIterator::new([Ok(data.clone())], data.schema());
        let dataset = Dataset::write(batches, "memory://", Some(write_params))
            .await
            .unwrap();

        assert_eq!(dataset.count_rows(None).await.unwrap(), 400);
        let projection = ProjectionRequest::from_sql(vec![
            ("foo", "i"),
            ("bar", "i*2"),
            ("_rowid", "_rowid"),
            ("_rowaddr", "_rowaddr"),
        ]);
        let values = dataset
            .take(&[10, 50, 100], projection.clone())
            .await
            .unwrap();
        let expected_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("foo", DataType::Int32, false),
            ArrowField::new("bar", DataType::Int32, false),
            ROW_ID_FIELD.clone(),
            ROW_ADDR_FIELD.clone(),
        ]));
        assert_eq!(
            RecordBatch::try_new(
                expected_schema,
                vec![
                    Arc::new(Int32Array::from_iter_values([10, 50, 100])),
                    Arc::new(Int32Array::from_iter_values([20, 100, 200])),
                    Arc::new(UInt64Array::from_iter_values([10, 50, 100])),
                    Arc::new(UInt64Array::from_iter_values([10, 4294967296, 8589934592])),
                ],
            )
            .unwrap(),
            values
        );

        let values2 = dataset.take_rows(&[10, 50, 100], projection).await.unwrap();
        assert_eq!(values, values2);
    }

    #[rstest]
    #[tokio::test]
    async fn test_take_rowid_rowaddr_with_projection_enable_stable_row_ids_projection_from_columns(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let data = test_batch(0..400);
        let write_params = WriteParams {
            data_storage_version: Some(data_storage_version),
            enable_stable_row_ids: true,
            max_rows_per_file: 50,
            ..Default::default()
        };
        let batches = RecordBatchIterator::new([Ok(data.clone())], data.schema());
        let dataset = Dataset::write(batches, "memory://", Some(write_params))
            .await
            .unwrap();

        assert_eq!(dataset.count_rows(None).await.unwrap(), 400);
        let projection =
            ProjectionRequest::from_columns(["_rowid", "_rowaddr", "i"], dataset.schema());

        let values = dataset
            .take(&[10, 50, 100], projection.clone())
            .await
            .unwrap();
        let expected_schema = Arc::new(ArrowSchema::new(vec![
            ROW_ID_FIELD.clone(),
            ROW_ADDR_FIELD.clone(),
            ArrowField::new("i", DataType::Int32, false),
        ]));
        assert_eq!(
            RecordBatch::try_new(
                expected_schema.clone(),
                vec![
                    Arc::new(UInt64Array::from_iter_values([10, 50, 100])),
                    Arc::new(UInt64Array::from_iter_values([10, 4294967296, 8589934592])),
                    Arc::new(Int32Array::from_iter_values([10, 50, 100])),
                ],
            )
            .unwrap(),
            values
        );

        let values2 = dataset
            .take_rows(&[10, 50, 100], projection.clone())
            .await
            .unwrap();
        assert_eq!(values, values2);

        let values3 = dataset
            .take(&[50, 100, 10], projection.clone())
            .await
            .unwrap();
        assert_eq!(
            RecordBatch::try_new(
                expected_schema,
                vec![
                    Arc::new(UInt64Array::from_iter_values([50, 100, 10])),
                    Arc::new(UInt64Array::from_iter_values([4294967296, 8589934592, 10])),
                    Arc::new(Int32Array::from_iter_values([50, 100, 10])),
                ],
            )
            .unwrap(),
            values3
        );
        let values4 = dataset.take_rows(&[50, 100, 10], projection).await.unwrap();
        assert_eq!(values3, values4);
    }

    #[rstest]
    #[tokio::test]
    async fn test_take_rowid_rowaddr_with_projection_disable_stable_row_ids_projection_from_sql(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let data = test_batch(0..400);
        let write_params = WriteParams {
            data_storage_version: Some(data_storage_version),
            enable_stable_row_ids: false,
            max_rows_per_file: 50,
            ..Default::default()
        };
        let batches = RecordBatchIterator::new([Ok(data.clone())], data.schema());
        let dataset = Dataset::write(batches, "memory://", Some(write_params))
            .await
            .unwrap();

        assert_eq!(dataset.count_rows(None).await.unwrap(), 400);
        let projection = ProjectionRequest::from_sql(vec![
            ("foo", "i"),
            ("bar", "i*2"),
            ("_rowid", "_rowid"),
            ("_rowaddr", "_rowaddr"),
        ]);
        let values = dataset
            .take(&[10, 50, 100], projection.clone())
            .await
            .unwrap();
        let expected_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("foo", DataType::Int32, false),
            ArrowField::new("bar", DataType::Int32, false),
            ROW_ID_FIELD.clone(),
            ROW_ADDR_FIELD.clone(),
        ]));
        assert_eq!(
            RecordBatch::try_new(
                expected_schema,
                vec![
                    Arc::new(Int32Array::from_iter_values([10, 50, 100])),
                    Arc::new(Int32Array::from_iter_values([20, 100, 200])),
                    Arc::new(UInt64Array::from_iter_values([10, 4294967296, 8589934592])),
                    Arc::new(UInt64Array::from_iter_values([10, 4294967296, 8589934592])),
                ],
            )
            .unwrap(),
            values
        );

        let values2 = dataset
            .take_rows(&[10, 4294967296, 8589934592], projection)
            .await
            .unwrap();
        assert_eq!(values, values2);
    }

    #[rstest]
    #[tokio::test]
    async fn test_take_rowid_rowaddr_with_projection_disable_stable_row_ids_projection_from_columns(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let data = test_batch(0..400);
        let write_params = WriteParams {
            data_storage_version: Some(data_storage_version),
            enable_stable_row_ids: false,
            max_rows_per_file: 50,
            ..Default::default()
        };
        let batches = RecordBatchIterator::new([Ok(data.clone())], data.schema());
        let dataset = Dataset::write(batches, "memory://", Some(write_params))
            .await
            .unwrap();

        assert_eq!(dataset.count_rows(None).await.unwrap(), 400);

        let projection =
            ProjectionRequest::from_columns(["_rowid", "_rowaddr", "i"], dataset.schema());
        let values = dataset
            .take(&[10, 50, 100], projection.clone())
            .await
            .unwrap();

        let expected_schema = Arc::new(ArrowSchema::new(vec![
            ROW_ID_FIELD.clone(),
            ROW_ADDR_FIELD.clone(),
            ArrowField::new("i", DataType::Int32, false),
        ]));

        assert_eq!(
            RecordBatch::try_new(
                expected_schema.clone(),
                vec![
                    Arc::new(UInt64Array::from_iter_values([10, 4294967296, 8589934592])),
                    Arc::new(UInt64Array::from_iter_values([10, 4294967296, 8589934592])),
                    Arc::new(Int32Array::from_iter_values([10, 50, 100])),
                ],
            )
            .unwrap(),
            values
        );
        let values2 = dataset
            .take_rows(&[10, 4294967296, 8589934592], projection.clone())
            .await
            .unwrap();
        assert_eq!(values, values2);

        let values3 = dataset
            .take(&[50, 100, 10], projection.clone())
            .await
            .unwrap();
        assert_eq!(
            RecordBatch::try_new(
                expected_schema,
                vec![
                    Arc::new(UInt64Array::from_iter_values([4294967296, 8589934592, 10])),
                    Arc::new(UInt64Array::from_iter_values([4294967296, 8589934592, 10])),
                    Arc::new(Int32Array::from_iter_values([50, 100, 10])),
                ],
            )
            .unwrap(),
            values3
        );
        let values4 = dataset
            .take_rows(&[4294967296, 8589934592, 10], projection)
            .await
            .unwrap();
        assert_eq!(values3, values4);
    }

    #[rstest]
    #[tokio::test]
    async fn test_take_rows_out_of_bound(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        // a dataset with 1 fragment and 400 rows
        let test_ds = TestVectorDataset::new(data_storage_version, false)
            .await
            .unwrap();
        let ds = test_ds.dataset;

        // take the last row of first fragment
        // this triggers the contiguous branch
        let indices = &[(1 << 32) - 1];
        let fut = require_send(ds.take_rows(indices, ds.schema().clone()));
        let err = fut.await.unwrap_err();
        assert!(
            err.to_string().contains("Invalid read params"),
            "{}",
            err.to_string()
        );

        // this triggers the sorted branch, but not contiguous
        let indices = &[(1 << 32) - 3, (1 << 32) - 1];
        let err = ds
            .take_rows(indices, ds.schema().clone())
            .await
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("Invalid read params Indices(4294967293,4294967295)"),
            "{}",
            err.to_string()
        );

        // this triggers the catch all branch
        let indices = &[(1 << 32) - 1, (1 << 32) - 3];
        let err = ds
            .take_rows(indices, ds.schema().clone())
            .await
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("Invalid read params Indices(4294967293,4294967295)"),
            "{}",
            err.to_string()
        );

        // Repeated maximum offsets must reach bounds validation without overflowing.
        let err = ds
            .take_rows(&[u32::MAX as u64; 2], ds.schema().clone())
            .await
            .unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "{err}");
        assert!(err.to_string().contains("Invalid read params"), "{err}");
        assert!(err.to_string().contains("4294967295"), "{err}");
    }

    #[rstest]
    #[case::contiguous(&[(2, 0)])]
    #[case::sorted(&[(0, 0), (2, 0)])]
    #[case::all_missing_sorted(&[(2, 0), (3, 0)])]
    #[case::all_missing_unsorted(&[(3, 0), (2, 0)])]
    #[case::all_missing_duplicates(&[(3, 0), (2, 0), (3, 0)])]
    #[case::partially_missing_unsorted(&[(2, 0), (1, 0)])]
    #[tokio::test]
    async fn test_take_rows_missing_fragments(
        #[case] addresses: &[(u32, u32)],
        #[values(false, true)] with_row_address: bool,
    ) {
        let data = test_batch(0..4);
        let dataset = Arc::new(
            Dataset::write(
                RecordBatchIterator::new([Ok(data.clone())], data.schema()),
                "memory://",
                Some(WriteParams {
                    max_rows_per_file: 2,
                    enable_stable_row_ids: true,
                    data_storage_version: Some(LanceFileVersion::Stable),
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );
        assert_eq!(dataset.get_fragments().len(), 2);
        let projection = ProjectionRequest::from(dataset.schema().clone())
            .into_projection_plan(dataset.clone())
            .unwrap();
        let addresses = addresses
            .iter()
            .map(|(fragment_id, offset)| {
                u64::from(RowAddress::new_from_parts(*fragment_id, *offset))
            })
            .collect();
        let error = TakeBuilder::try_new_from_addresses(dataset, addresses, Arc::new(projection))
            .unwrap()
            .with_row_address(with_row_address)
            .execute()
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        let message = error.to_string();
        assert!(
            message.contains("non-existent fragment: 2")
                || message.contains("Unknown fragment id 2"),
            "{message}"
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_take_rows(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let data = test_batch(0..400);
        let write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        let batches = RecordBatchIterator::new([Ok(data.clone())], data.schema());
        let mut dataset = Dataset::write(batches, "memory://", Some(write_params))
            .await
            .unwrap();

        assert_eq!(dataset.count_rows(None).await.unwrap(), 400);
        let projection = Schema::try_from(data.schema().as_ref()).unwrap();
        let indices = &[
            5_u64 << 32,        // 200
            (4_u64 << 32) + 39, // 199
            39,                 // 39
            1_u64 << 32,        // 40
            (2_u64 << 32) + 20, // 100
        ];
        let values = dataset
            .take_rows(indices, projection.clone())
            .await
            .unwrap();
        assert_eq!(
            RecordBatch::try_new(
                data.schema(),
                vec![
                    Arc::new(Int32Array::from_iter_values([200, 199, 39, 40, 100])),
                    Arc::new(StringArray::from_iter_values(
                        [200, 199, 39, 40, 100].iter().map(|v| format!("str-{v}"))
                    )),
                ],
            )
            .unwrap(),
            values
        );

        // Delete some rows from a fragment
        dataset.delete("i in (199, 100)").await.unwrap();
        dataset.validate().await.unwrap();
        let values = dataset
            .take_rows(indices, projection.clone())
            .await
            .unwrap();
        assert_eq!(
            RecordBatch::try_new(
                data.schema(),
                vec![
                    Arc::new(Int32Array::from_iter_values([200, 39, 40])),
                    Arc::new(StringArray::from_iter_values(
                        [200, 39, 40].iter().map(|v| format!("str-{v}"))
                    )),
                ],
            )
            .unwrap(),
            values
        );

        // Take an empty selection.
        let values = dataset.take_rows(&[], projection).await.unwrap();
        assert_eq!(RecordBatch::new_empty(data.schema()), values);
    }

    #[tokio::test]
    async fn test_take_rows_with_io_parallelism() {
        let data = test_batch(0..400);
        let write_params = WriteParams {
            max_rows_per_file: 40,
            ..Default::default()
        };
        let batches = RecordBatchIterator::new([Ok(data.clone())], data.schema());
        let dataset = Arc::new(
            Dataset::write(batches, "memory://", Some(write_params))
                .await
                .unwrap(),
        );
        let projection = Schema::try_from(data.schema().as_ref()).unwrap();
        // Rows from four fragments, out of order.
        let indices = [5_u64 << 32, (4_u64 << 32) + 39, 39, (2_u64 << 32) + 20];

        // One request at a time across all fragments still returns every row.
        let expected = dataset
            .take_rows(&indices, projection.clone())
            .await
            .unwrap();
        let values = dataset
            .take_builder(&indices, projection.clone())
            .unwrap()
            .with_io_parallelism(1)
            .execute()
            .await
            .unwrap();
        assert_eq!(expected, values);

        // The public logical-offset API takes the same path and also handles an
        // empty selection without constructing a scheduler.
        let logical_indices = [399, 0, 199, 40];
        let expected = dataset
            .take(&logical_indices, projection.clone())
            .await
            .unwrap();
        let values = dataset
            .take_with_io_parallelism(&logical_indices, projection.clone(), 1)
            .await
            .unwrap();
        assert_eq!(expected, values);
        // Large caps (including u32::MAX) must work without over-allocating.
        let values = dataset
            .take_with_io_parallelism(&logical_indices, projection.clone(), u32::MAX)
            .await
            .unwrap();
        assert_eq!(expected, values);

        // Sorted multi-fragment selections take the ordered path.
        let sorted_logical_indices = [0, 40, 199, 399];
        let sorted_expected = dataset
            .take(&sorted_logical_indices, projection.clone())
            .await
            .unwrap();
        let sorted_values = dataset
            .take_with_io_parallelism(&sorted_logical_indices, projection.clone(), 1)
            .await
            .unwrap();
        assert_eq!(sorted_expected, sorted_values);

        let empty = dataset
            .take_with_io_parallelism(&[], projection.clone(), 1)
            .await
            .unwrap();
        assert_eq!(empty, dataset.take(&[], projection.clone()).await.unwrap());

        let error = dataset
            .take_with_io_parallelism(&indices, projection.clone(), 0)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(error.to_string().contains("io_parallelism"), "{error}");

        let error = dataset
            .take_builder(&indices, projection.clone())
            .unwrap()
            .with_io_parallelism(0)
            .execute()
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(error.to_string().contains("io_parallelism"), "{error}");

        let error = dataset
            .take_builder(&[], projection)
            .unwrap()
            .with_io_parallelism(0)
            .execute()
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(error.to_string().contains("io_parallelism"), "{error}");
    }

    #[tokio::test]
    async fn test_take_rows_enforces_io_parallelism_across_fragments() {
        let data = test_batch(0..400);
        let batches = RecordBatchIterator::new([Ok(data.clone())], data.schema());
        let mut dataset = Dataset::write(
            batches,
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 40,
                data_storage_version: Some(LanceFileVersion::Stable),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let active_reads = Arc::new(AtomicUsize::new(0));
        let peak_reads = Arc::new(AtomicUsize::new(0));

        // Increment immediately before the artificial delay and decrement after
        // it. The interval therefore exposes how many data-file GETs the take's
        // scheduler allowed to proceed concurrently.
        let mut completed_policy = ProxyObjectStorePolicy::new();
        let completed_active = active_reads.clone();
        completed_policy.set_before_policy(
            "complete_take_data_read",
            Arc::new(move |method, path| {
                if method == "get_opts" && path.as_ref().contains("data/") {
                    completed_active.fetch_sub(1, Ordering::SeqCst);
                }
                Ok(())
            }),
        );
        let delayed = Arc::new(ThrottledStore::new(
            Arc::new(ProxyObjectStore::new(
                dataset.table_state.object_store.inner.clone(),
                Arc::new(Mutex::new(completed_policy)),
            )),
            ThrottleConfig {
                wait_get_per_call: Duration::from_millis(10),
                ..Default::default()
            },
        ));
        let mut started_policy = ProxyObjectStorePolicy::new();
        let started_active = active_reads.clone();
        let started_peak = peak_reads.clone();
        started_policy.set_before_policy(
            "start_take_data_read",
            Arc::new(move |method, path| {
                if method == "get_opts" && path.as_ref().contains("data/") {
                    let active = started_active.fetch_add(1, Ordering::SeqCst) + 1;
                    started_peak.fetch_max(active, Ordering::SeqCst);
                }
                Ok(())
            }),
        );
        Arc::make_mut(&mut dataset.table_state.object_store).inner = Arc::new(
            ProxyObjectStore::new(delayed, Arc::new(Mutex::new(started_policy))),
        );

        let projection = Schema::try_from(data.schema().as_ref()).unwrap();
        // Logical offsets touch all ten fragments in reverse order.
        let indices = (0_u64..10)
            .rev()
            .map(|fragment| fragment * 40)
            .collect::<Vec<_>>();
        let fragment_fanout = dataset
            .table_state
            .object_store
            .io_parallelism()
            .min(indices.len());
        let uncapped = dataset.take(&indices, projection.clone()).await.unwrap();
        let uncapped_peak = peak_reads.load(Ordering::SeqCst);
        assert_eq!(active_reads.load(Ordering::SeqCst), 0);
        if fragment_fanout > 2 {
            assert!(
                uncapped_peak > 2,
                "uncapped take must exceed the tested cap, observed {uncapped_peak}"
            );
        } else {
            assert_eq!(uncapped_peak, fragment_fanout);
        }

        peak_reads.store(0, Ordering::SeqCst);
        let capped = dataset
            .take_with_io_parallelism(&indices, projection.clone(), 2)
            .await
            .unwrap();

        assert_eq!(capped, uncapped);
        assert_eq!(active_reads.load(Ordering::SeqCst), 0);
        assert_eq!(peak_reads.load(Ordering::SeqCst), fragment_fanout.min(2));

        // Sorted multi-fragment takes must enforce the same cap.
        let sorted_indices = (0_u64..10)
            .map(|fragment| fragment * 40)
            .collect::<Vec<_>>();
        let sorted_uncapped = dataset
            .take(&sorted_indices, projection.clone())
            .await
            .unwrap();
        peak_reads.store(0, Ordering::SeqCst);
        let sorted_capped = dataset
            .take_with_io_parallelism(&sorted_indices, projection, 2)
            .await
            .unwrap();

        assert_eq!(sorted_capped, sorted_uncapped);
        assert_eq!(active_reads.load(Ordering::SeqCst), 0);
        assert_eq!(peak_reads.load(Ordering::SeqCst), fragment_fanout.min(2));
    }

    #[tokio::test]
    async fn test_take_rows_enforces_io_parallelism_across_data_files_in_fragment() {
        let data = test_batch(0..40);
        let batches = RecordBatchIterator::new([Ok(data.clone())], data.schema());
        let mut dataset = Dataset::write(
            batches,
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 40,
                data_storage_version: Some(LanceFileVersion::Stable),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Add three extra column files to the same fragment so a single-fragment
        // take opens and reads four data files concurrently.
        for col_name in ["c1", "c2", "c3"] {
            let extra_schema = Arc::new(ArrowSchema::new(vec![
                ArrowField::new("i", DataType::Int32, false),
                ArrowField::new(col_name, DataType::Int32, false),
            ]));
            let extra_batch = RecordBatch::try_new(
                extra_schema.clone(),
                vec![
                    Arc::new(Int32Array::from_iter_values(0..40)),
                    Arc::new(Int32Array::from_iter_values(100..140)),
                ],
            )
            .unwrap();
            dataset
                .merge(
                    RecordBatchIterator::new([Ok(extra_batch)], extra_schema),
                    "i",
                    "i",
                )
                .await
                .unwrap();
        }
        assert_eq!(dataset.get_fragments().len(), 1);
        assert_eq!(dataset.get_fragments()[0].metadata().files.len(), 4);

        let active_reads = Arc::new(AtomicUsize::new(0));
        let peak_reads = Arc::new(AtomicUsize::new(0));

        let mut completed_policy = ProxyObjectStorePolicy::new();
        let completed_active = active_reads.clone();
        completed_policy.set_before_policy(
            "complete_single_frag_data_read",
            Arc::new(move |method, path| {
                if method == "get_opts" && path.as_ref().contains("data/") {
                    completed_active.fetch_sub(1, Ordering::SeqCst);
                }
                Ok(())
            }),
        );
        let delayed = Arc::new(ThrottledStore::new(
            Arc::new(ProxyObjectStore::new(
                dataset.table_state.object_store.inner.clone(),
                Arc::new(Mutex::new(completed_policy)),
            )),
            ThrottleConfig {
                wait_get_per_call: Duration::from_millis(20),
                ..Default::default()
            },
        ));
        let mut started_policy = ProxyObjectStorePolicy::new();
        let started_active = active_reads.clone();
        let started_peak = peak_reads.clone();
        started_policy.set_before_policy(
            "start_single_frag_data_read",
            Arc::new(move |method, path| {
                if method == "get_opts" && path.as_ref().contains("data/") {
                    let active = started_active.fetch_add(1, Ordering::SeqCst) + 1;
                    started_peak.fetch_max(active, Ordering::SeqCst);
                }
                Ok(())
            }),
        );
        Arc::make_mut(&mut dataset.table_state.object_store).inner = Arc::new(
            ProxyObjectStore::new(delayed, Arc::new(Mutex::new(started_policy))),
        );

        let projection = dataset.schema().clone();
        let indices = [20_u64, 0];
        let uncapped = dataset.take(&indices, projection.clone()).await.unwrap();
        let uncapped_peak = peak_reads.load(Ordering::SeqCst);
        assert_eq!(active_reads.load(Ordering::SeqCst), 0);
        assert!(
            uncapped_peak > 2,
            "uncapped multi-file single-fragment take must exceed cap 2, observed {uncapped_peak}"
        );

        peak_reads.store(0, Ordering::SeqCst);
        let capped_two = dataset
            .take_with_io_parallelism(&indices, projection.clone(), 2)
            .await
            .unwrap();
        assert_eq!(capped_two, uncapped);
        assert_eq!(active_reads.load(Ordering::SeqCst), 0);
        assert_eq!(peak_reads.load(Ordering::SeqCst), 2);

        peak_reads.store(0, Ordering::SeqCst);
        let capped_one = dataset
            .take_with_io_parallelism(&indices, projection, 1)
            .await
            .unwrap();
        assert_eq!(capped_one, uncapped);
        assert_eq!(active_reads.load(Ordering::SeqCst), 0);
        assert_eq!(peak_reads.load(Ordering::SeqCst), 1);
    }

    #[rstest]
    #[tokio::test]
    async fn take_scan_dataset(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        use arrow::datatypes::Int32Type;

        let data = test_batch(1..5);
        let write_params = WriteParams {
            max_rows_per_group: 2,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        let batches = RecordBatchIterator::new([Ok(data.clone())], data.schema());
        let dataset = Dataset::write(batches, "memory://", Some(write_params))
            .await
            .unwrap();

        let projection = Arc::new(dataset.schema().project(&["i"]).unwrap());
        let ranges = [0_u64..3, 1..4, 0..1];
        let range_stream = futures::stream::iter(ranges).map(Ok).boxed();
        let results = dataset
            .take_scan(range_stream, projection.clone(), 10)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let expected_schema = projection.as_ref().into();
        for batch in &results {
            assert_eq!(batch.schema().as_ref(), &expected_schema);
        }
        assert_eq!(results.len(), 3);
        assert_eq!(
            results[0].column(0).as_primitive::<Int32Type>().values(),
            &[1, 2, 3],
        );
        assert_eq!(
            results[1].column(0).as_primitive::<Int32Type>().values(),
            &[2, 3, 4],
        );
        assert_eq!(
            results[2].column(0).as_primitive::<Int32Type>().values(),
            &[1],
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_take_scan_readahead(
        #[values(0, 1)] batch_readahead: usize,
        #[values(false, true)] empty_input: bool,
    ) {
        let data = test_batch(0..4);
        let dataset = Dataset::write(
            RecordBatchIterator::new([Ok(data.clone())], data.schema()),
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 2,
                data_storage_version: Some(LanceFileVersion::Stable),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.get_fragments().len(), 2);
        let ranges = if empty_input { vec![] } else { vec![Ok(0..3)] };
        let mut stream = dataset.take_scan(
            futures::stream::iter(ranges).boxed(),
            Arc::new(dataset.schema().clone()),
            batch_readahead,
        );
        assert_eq!(
            stream.schema().as_ref(),
            &ArrowSchema::from(dataset.schema())
        );

        if batch_readahead == 0 {
            let error = tokio::time::timeout(std::time::Duration::from_millis(100), stream.next())
                .await
                .expect("zero readahead must not hang")
                .unwrap()
                .unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
            assert!(
                error
                    .to_string()
                    .contains("batch_readahead must be greater than 0, got 0"),
                "{error}"
            );
        } else if !empty_input {
            assert_eq!(stream.next().await.unwrap().unwrap().num_rows(), 3);
        }
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(100), stream.next())
                .await
                .expect("take stream must terminate")
                .is_none()
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_take_rows_with_row_ids(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let data = test_batch(0..8);
        let write_params = WriteParams {
            max_rows_per_group: 2,
            data_storage_version: Some(data_storage_version),
            enable_stable_row_ids: true,
            ..Default::default()
        };
        let batches = RecordBatchIterator::new([Ok(data.clone())], data.schema());
        let mut dataset = Dataset::write(batches, "memory://", Some(write_params))
            .await
            .unwrap();

        dataset.delete("i in (1, 2, 3, 7)").await.unwrap();

        let indices = &[0, 4, 6, 5];
        let result = dataset
            .take_rows(indices, dataset.schema().clone())
            .await
            .unwrap();
        assert_eq!(
            RecordBatch::try_new(
                data.schema(),
                vec![
                    Arc::new(Int32Array::from_iter_values(
                        indices.iter().map(|x| *x as i32)
                    )),
                    Arc::new(StringArray::from_iter_values(
                        indices.iter().map(|v| format!("str-{v}"))
                    )),
                ],
            )
            .unwrap(),
            result
        );
    }

    /// An update keeps the old copy of a row under a deletion vector and writes
    /// its id again in a new fragment. Ignoring deletions must still resolve
    /// the id to the live copy, which takes that fragment's deletion vector.
    #[tokio::test]
    async fn test_take_rows_ignore_deletions_after_update() {
        let data = test_batch(0..8);
        let write_params = WriteParams {
            enable_stable_row_ids: true,
            ..Default::default()
        };
        let batches = RecordBatchIterator::new([Ok(data.clone())], data.schema());
        let dataset = Dataset::write(batches, "memory://", Some(write_params))
            .await
            .unwrap();
        let dataset = UpdateBuilder::new(Arc::new(dataset))
            .update_where("i = 1")
            .unwrap()
            .set("s", "'updated'")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap()
            .new_dataset;

        for ignore_deletions in [false, true] {
            let point = dataset
                .take_builder(&[1], dataset.schema().clone())
                .unwrap()
                .with_ignore_deletions(ignore_deletions)
                .execute()
                .await
                .unwrap();
            assert_eq!(point["s"].as_string::<i32>().value(0), "updated");
        }

        let checked = dataset
            .take_rows(&[1, 2], dataset.schema().clone())
            .await
            .unwrap();
        assert_eq!(checked["s"].as_string::<i32>().value(0), "updated");
        let ignored = dataset
            .take_builder(&[1, 2], dataset.schema().clone())
            .unwrap()
            .with_ignore_deletions(true)
            .execute()
            .await
            .unwrap();
        assert_eq!(ignored, checked);
    }

    /// Ignoring deletions resolves and reads the rows without touching a
    /// deletion file, and hands back the deleted rows with their stored values.
    #[tokio::test]
    async fn test_take_rows_ignore_deletions() {
        let data = test_batch(0..8);
        let write_params = WriteParams {
            max_rows_per_group: 2,
            enable_stable_row_ids: true,
            ..Default::default()
        };
        let batches = RecordBatchIterator::new([Ok(data.clone())], data.schema());
        let mut dataset = Dataset::write(batches, "memory://", Some(write_params))
            .await
            .unwrap();
        dataset.delete("i in (1, 7)").await.unwrap();
        let dataset = Arc::new(dataset);

        let reads_deletion_file = |dataset: &Dataset| {
            dataset
                .table_state
                .object_store
                .io_stats_incremental()
                .requests
                .iter()
                .any(|request| request.path.as_ref().contains("_deletions"))
        };
        dataset.table_state.object_store.io_stats_incremental();

        let ignored = dataset
            .take_builder(&[0, 1, 7, 4], dataset.schema().clone())
            .unwrap()
            .with_ignore_deletions(true)
            .execute()
            .await
            .unwrap();
        assert_eq!(
            ignored["i"]
                .as_primitive::<arrow_array::types::Int32Type>()
                .values()
                .as_ref(),
            &[0, 1, 7, 4],
            "deleted rows 1 and 7 come back with their stored values"
        );
        assert!(
            !reads_deletion_file(&dataset),
            "ignoring deletions must not read a deletion file"
        );

        let checked = dataset
            .take_rows(&[0, 1, 7, 4], dataset.schema().clone())
            .await
            .unwrap();
        assert_eq!(
            checked["i"]
                .as_primitive::<arrow_array::types::Int32Type>()
                .values()
                .as_ref(),
            &[0, 4]
        );
        assert!(reads_deletion_file(&dataset));
    }
}
