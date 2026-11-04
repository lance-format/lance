// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The public entry point for rewriting batches that carry Blob v2 columns.
//!
//! A Blob v2 value does not live in the row. The row holds a descriptor, and the
//! payload lives in separate storage reached through it. Any operation that
//! rewrites rows into new fragments therefore has to decide what happens to that
//! payload: keep the reference and publish the descriptor again, or read the bytes
//! back and write them into the new fragment.
//!
//! Getting that decision wrong is silent. The batch still has the right number of
//! rows and the right column names, so a rewrite that quietly dropped a reference
//! looks exactly like one that preserved it until something later tries to read the
//! payload. [`BlobRewritePlan`] exists so that callers outside this crate do not
//! have to make, or even know about, that decision.

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::{
    DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema, SchemaRef,
};
use lance_core::ROW_ADDR;
use lance_file::version::ConcreteFileVersion;
use lance_table::feature_flags::managed_blobs_enabled;

use super::rewrite::BlobV2BatchRewritePlan;
use crate::dataset::Dataset;
use crate::{Error, Result};

/// What a plan does to each batch.
enum Rewrite {
    /// The dataset has no Blob v2 column reachable from this batch, so the batch
    /// is already in its final form. Only the row address column is dropped, when
    /// the caller asked for that.
    Passthrough { drop_row_addresses: bool },
    /// At least one Blob v2 column needs rewriting.
    BlobV2(BlobV2BatchRewritePlan),
}

/// How a batch of rows carrying Blob v2 columns is rewritten for a new fragment.
///
/// Build one plan per `(dataset, input batch schema)` pair and reuse it for every
/// batch with that schema; constructing it inspects the schema, so doing it per
/// batch is wasted work rather than wrong.
///
/// # Which rewrite is chosen
///
/// The plan derives this from the dataset, and deliberately exposes no switch for
/// it. Preserving a managed reference is only sound when the destination file
/// format can represent a prepared blob and the dataset has already opted in to
/// managed blobs; otherwise the payload must be materialized. Both conditions are
/// properties of the dataset, not preferences of the caller, so a caller that could
/// override them could only use that power to corrupt its own output.
///
/// # The batches must come from this dataset
///
/// Blob payloads and managed descriptors are resolved by row address against the
/// dataset the plan was built from. Transforming a batch whose row addresses mean
/// something else — read from a different dataset, or from a different version of
/// this one where those addresses have been rewritten — produces wrong payloads
/// rather than an error, because a row address carries nothing that identifies its
/// origin.
///
/// Holding the dataset lets [`Self::transform`] resolve addresses without the
/// caller passing it back, and it rules out pairing the plan with a *handle* to
/// another dataset. It cannot establish where the rows came from: that remains the
/// caller's precondition. [`Self::transform`] does check that each batch matches
/// the schema the plan was built for, which catches a mismatched batch but not a
/// same-shaped batch from elsewhere.
///
/// # A preserved reference does not own its payload
///
/// When the managed payload is preserved rather than materialized, the output
/// holds a reference to an object under the *source* dataset's blob storage. The
/// plan does not retain that object. Until the destination's references are no
/// longer needed, the caller must keep the source versions they name alive, which
/// means not advancing [`cleanup_old_versions`] past them. A destination in
/// another dataset must additionally register the source as a base, because with
/// primary base-less storage the preserved reference is a plain `Managed` one that
/// resolves only within the dataset that holds it. This is the lifetime contract
/// [`preserve_managed_descriptors`] documents for same-dataset rewrites, stated
/// here for callers outside this crate.
///
/// [`cleanup_old_versions`]: crate::dataset::cleanup::cleanup_old_versions
/// [`preserve_managed_descriptors`]: crate::dataset::preserve_managed_descriptors
///
/// # Example
///
/// ```no_run
/// # use std::sync::Arc;
/// # use arrow_array::RecordBatch;
/// # use lance::dataset::{BlobRewritePlan, Dataset};
/// # async fn example(
/// #     dataset: Arc<Dataset>,
/// #     batches: Vec<RecordBatch>,
/// # ) -> lance::Result<()> {
/// let first = batches.first().expect("at least one batch");
/// let plan = BlobRewritePlan::try_new(dataset, first.schema().as_ref(), false)?;
///
/// // The output schema is known before any batch is transformed, so it can be
/// // used to open the writer.
/// let output_schema = plan.output_schema().clone();
///
/// for batch in batches {
///     let rewritten = plan.transform(batch).await?;
///     assert_eq!(rewritten.schema(), output_schema);
/// }
/// # Ok(())
/// # }
/// ```
pub struct BlobRewritePlan {
    dataset: Arc<Dataset>,
    input_schema: SchemaRef,
    output_schema: SchemaRef,
    rewrite: Rewrite,
}

impl std::fmt::Debug for BlobRewritePlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlobRewritePlan")
            .field("uri", &self.dataset.uri())
            .field(
                "rewrites_blob_v2",
                &matches!(self.rewrite, Rewrite::BlobV2(_)),
            )
            .field("input_schema", &self.input_schema)
            .field("output_schema", &self.output_schema)
            .finish()
    }
}

impl BlobRewritePlan {
    /// Plan the rewrite of batches shaped like `input_schema`, read from `dataset`.
    ///
    /// `retain_row_addresses` keeps the row address column in the output.
    ///
    /// Rewriting a Blob v2 column needs row addresses in the *input*, because that
    /// is how a descriptor is resolved to its payload. So when the dataset has a
    /// Blob v2 column reachable from this schema, `input_schema` must contain the
    /// row address column and this returns an error if it does not. A schema with
    /// nothing to rewrite has no such requirement, and `retain_row_addresses` then
    /// only decides whether an existing row address column is carried through.
    pub fn try_new(
        dataset: Arc<Dataset>,
        input_schema: &ArrowSchema,
        retain_row_addresses: bool,
    ) -> Result<Self> {
        let input_schema = Arc::new(input_schema.clone());
        let preserve_managed = Self::can_preserve_managed(&dataset)?;

        // Only the plan itself knows which columns it would touch, and it refuses to
        // be built without a row address column. So ask it on a schema that is the
        // caller's with a row address column added when missing: one schema clone and
        // one plan construction, no I/O.
        let probe_schema = if input_schema.column_with_name(ROW_ADDR).is_none() {
            let mut fields = input_schema.fields().to_vec();
            fields.push(Arc::new(ArrowField::new(
                ROW_ADDR,
                ArrowDataType::UInt64,
                false,
            )));
            Arc::new(ArrowSchema::new_with_metadata(
                fields,
                input_schema.metadata().clone(),
            ))
        } else {
            input_schema.clone()
        };
        let rewrites_blobs = BlobV2BatchRewritePlan::try_new(
            dataset.schema(),
            probe_schema.as_ref(),
            retain_row_addresses,
            preserve_managed,
        )?
        .rewrites_anything();

        if !rewrites_blobs {
            let drop_row_addresses = !retain_row_addresses;
            let output_schema = if drop_row_addresses {
                Arc::new(ArrowSchema::new_with_metadata(
                    input_schema
                        .fields()
                        .iter()
                        .filter(|field| field.name() != ROW_ADDR)
                        .map(Arc::clone)
                        .collect::<Vec<_>>(),
                    input_schema.metadata().clone(),
                ))
            } else {
                input_schema.clone()
            };
            return Ok(Self {
                dataset,
                input_schema,
                output_schema,
                rewrite: Rewrite::Passthrough { drop_row_addresses },
            });
        }

        if input_schema.column_with_name(ROW_ADDR).is_none() {
            return Err(Error::invalid_input(format!(
                "rewriting a Blob v2 column needs the {ROW_ADDR} column in the input batch, \
                 because a blob descriptor is resolved to its payload by row address; \
                 the given schema has only: {:?}",
                input_schema
                    .fields()
                    .iter()
                    .map(|field| field.name())
                    .collect::<Vec<_>>()
            )));
        }

        let inner = BlobV2BatchRewritePlan::try_new(
            dataset.schema(),
            input_schema.as_ref(),
            retain_row_addresses,
            preserve_managed,
        )?;
        let output_schema = inner.output_schema().clone();
        Ok(Self {
            dataset,
            input_schema,
            output_schema,
            rewrite: Rewrite::BlobV2(inner),
        })
    }

    /// The schema every [`Self::transform`] output carries.
    ///
    /// This is known at construction because the rewrite each Blob v2 column takes
    /// is decided then, and that choice determines the column's output type. Open
    /// the destination writer with this schema rather than with the input one.
    pub fn output_schema(&self) -> &SchemaRef {
        &self.output_schema
    }

    /// Rewrite one batch, resolving Blob v2 columns against the plan's dataset.
    ///
    /// The batch must have the schema this plan was built for; the rewrite reads
    /// columns by position, so a batch of a different shape is rejected rather than
    /// silently misread. See the note on this type about the rows' origin, which
    /// this check cannot verify.
    ///
    /// Batches may be transformed concurrently. Depending on the rewrite chosen,
    /// this may read payload bytes from storage, so it is not a cheap in-memory
    /// projection.
    pub async fn transform(&self, batch: RecordBatch) -> Result<RecordBatch> {
        if batch.schema().fields() != self.input_schema.fields() {
            return Err(Error::invalid_input(format!(
                "this batch does not have the schema its plan was built for; expected \
                 columns {:?} but got {:?}",
                self.input_schema
                    .fields()
                    .iter()
                    .map(|field| field.name())
                    .collect::<Vec<_>>(),
                batch
                    .schema()
                    .fields()
                    .iter()
                    .map(|field| field.name())
                    .collect::<Vec<_>>()
            )));
        }
        match &self.rewrite {
            Rewrite::Passthrough { drop_row_addresses } => {
                if !drop_row_addresses {
                    return Ok(batch);
                }
                Ok(batch.project(
                    &(0..batch.num_columns())
                        .filter(|index| batch.schema().field(*index).name() != ROW_ADDR)
                        .collect::<Vec<_>>(),
                )?)
            }
            Rewrite::BlobV2(inner) => inner.transform_batch(&self.dataset, batch).await,
        }
    }

    /// Managed descriptors may be preserved only when the dataset's own file format
    /// can represent a prepared blob and managed blobs are already enabled on it.
    fn can_preserve_managed(dataset: &Arc<Dataset>) -> Result<bool> {
        let supports_prepared = matches!(
            dataset.manifest().data_storage_format.lance_file_format(),
            ConcreteFileVersion::V2_2 | ConcreteFileVersion::V2_3
        );
        Ok(supports_prepared && managed_blobs_enabled(dataset.schema())?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blob::{BlobArrayBuilder, BlobFieldOptions, blob_field_with_options};
    use crate::blob_field;
    use crate::dataset::blob::test_util::managed_blob_schema;
    use crate::dataset::cleanup::{CleanupPolicy, cleanup_old_versions};
    use crate::dataset::{WriteMode, WriteParams};
    use arrow_array::Array;
    use arrow_array::cast::AsArray;
    use arrow_array::{
        ArrayRef, LargeBinaryArray, LargeListArray, ListArray, RecordBatchIterator, StringArray,
        StructArray, UInt32Array, UInt64Array,
    };
    use arrow_buffer::{OffsetBuffer, ScalarBuffer};
    use arrow_schema::{DataType, Field, Fields, Schema};
    use futures::TryStreamExt;
    use lance_core::datatypes::{BLOB_V2_LOGICAL_TYPE, BlobKind, BlobV2Layout};
    use lance_core::utils::tempfile::TempStrDir;
    use lance_file::version::LanceFileVersion;
    use lance_table::format::BasePath;
    use std::num::NonZeroUsize;

    /// A dataset with no Blob v2 column anywhere in its schema.
    async fn plain_dataset(dir: &TempStrDir) -> Arc<Dataset> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt32, false),
            Field::new("count", DataType::UInt32, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from_iter_values(0..4)),
                Arc::new(UInt32Array::from_iter_values(10..14)),
            ],
        )
        .unwrap();
        Arc::new(
            Dataset::write(
                RecordBatchIterator::new([Ok(batch)], schema),
                dir,
                Some(WriteParams {
                    data_storage_version: Some(LanceFileVersion::V2_2),
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        )
    }

    /// A Blob v2 dataset with managed blobs off, split into one-row fragments so a
    /// scan yields more than one batch.
    async fn blob_v2_dataset(dir: &TempStrDir, rows: usize) -> Arc<Dataset> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt32, false),
            blob_field("blob", true),
        ]));
        let mut blobs = BlobArrayBuilder::new(rows);
        for row in 0..rows {
            blobs.push_bytes(vec![row as u8; 16]).unwrap();
        }
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from_iter_values(0..rows as u32)),
                blobs.finish().unwrap(),
            ],
        )
        .unwrap();
        let dataset = Arc::new(
            Dataset::write(
                RecordBatchIterator::new([Ok(batch)], schema),
                dir,
                Some(WriteParams {
                    data_storage_version: Some(LanceFileVersion::V2_2),
                    max_rows_per_file: 1,
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );
        assert!(!managed_blobs_enabled(dataset.schema()).unwrap());
        dataset
    }

    async fn scan_without_row_addresses(dataset: &Dataset) -> RecordBatch {
        let batch = dataset.scan().try_into_batch().await.unwrap();
        assert!(batch.schema().column_with_name(ROW_ADDR).is_none());
        batch
    }

    async fn scan_with_row_addresses(dataset: &Dataset) -> Vec<RecordBatch> {
        let mut scan = dataset.scan();
        scan.with_row_address();
        scan.try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap()
    }

    fn blob_output_fields(schema: &ArrowSchema, name: &str) -> Fields {
        let DataType::Struct(fields) = schema.field_with_name(name).unwrap().data_type() else {
            panic!("blob column '{name}' should be a struct in the output schema");
        };
        fields.clone()
    }

    #[tokio::test]
    async fn a_dataset_without_blob_v2_passes_every_batch_through_unchanged() {
        let dir = TempStrDir::default();
        let dataset = plain_dataset(&dir).await;
        let input = scan_without_row_addresses(&dataset).await;

        let plan =
            BlobRewritePlan::try_new(dataset.clone(), input.schema().as_ref(), false).unwrap();
        assert_eq!(plan.output_schema(), &input.schema());

        let output = plan.transform(input.clone()).await.unwrap();
        assert_eq!(output.schema(), input.schema());
        assert_eq!(output, input);
    }

    #[tokio::test]
    async fn a_missing_row_address_column_is_refused_only_when_a_blob_must_be_rewritten() {
        let blob_dir = TempStrDir::default();
        let blob_dataset = blob_v2_dataset(&blob_dir, 2).await;
        let blob_input = scan_without_row_addresses(&blob_dataset).await;
        let error =
            BlobRewritePlan::try_new(blob_dataset.clone(), blob_input.schema().as_ref(), false)
                .unwrap_err();
        assert!(
            matches!(error, Error::InvalidInput { .. }),
            "unexpected error variant: {error:?}"
        );
        assert!(error.to_string().contains(ROW_ADDR), "{error}");

        let plain_dir = TempStrDir::default();
        let plain = plain_dataset(&plain_dir).await;
        let plain_input = scan_without_row_addresses(&plain).await;
        BlobRewritePlan::try_new(plain.clone(), plain_input.schema().as_ref(), false).unwrap();
    }

    #[tokio::test]
    async fn the_output_schema_is_known_before_any_batch_and_every_batch_matches_it() {
        let dir = TempStrDir::default();
        let dataset = blob_v2_dataset(&dir, 4).await;
        let batches = scan_with_row_addresses(&dataset).await;
        assert!(
            batches.len() >= 2,
            "need at least two batches to show the schema is stable across them"
        );

        let plan =
            BlobRewritePlan::try_new(dataset.clone(), batches[0].schema().as_ref(), false).unwrap();
        let output_schema = plan.output_schema().clone();

        for batch in batches {
            let rewritten = plan.transform(batch).await.unwrap();
            assert_eq!(rewritten.schema(), output_schema);
        }
    }

    #[tokio::test]
    async fn a_batch_that_does_not_match_the_plans_input_schema_is_refused() {
        let dir = TempStrDir::default();
        let dataset = blob_v2_dataset(&dir, 1).await;
        let batch = scan_with_row_addresses(&dataset).await.remove(0);
        let plan =
            BlobRewritePlan::try_new(dataset.clone(), batch.schema().as_ref(), false).unwrap();

        // Dropping the blob column leaves the row address column at a different
        // position, which positional access would misread rather than reject.
        let blob_index = batch.schema().index_of("blob").unwrap();
        let projected = batch
            .project(
                &(0..batch.num_columns())
                    .filter(|index| *index != blob_index)
                    .collect::<Vec<_>>(),
            )
            .unwrap();

        let error = plan.transform(projected).await.unwrap_err();
        assert!(
            matches!(error, Error::InvalidInput { .. }),
            "unexpected error variant: {error:?}"
        );
        let message = error.to_string();
        for name in batch.schema().fields().iter().map(|field| field.name()) {
            assert!(message.contains(name.as_str()), "{message}");
        }
    }

    #[tokio::test]
    async fn retain_row_addresses_controls_only_the_output_column() {
        let dir = TempStrDir::default();
        let dataset = blob_v2_dataset(&dir, 2).await;
        let input_schema = scan_with_row_addresses(&dataset).await[0].schema();

        let dropped = BlobRewritePlan::try_new(dataset.clone(), input_schema.as_ref(), false)
            .unwrap()
            .output_schema()
            .clone();
        let retained = BlobRewritePlan::try_new(dataset.clone(), input_schema.as_ref(), true)
            .unwrap()
            .output_schema()
            .clone();

        assert!(dropped.column_with_name(ROW_ADDR).is_none());
        assert!(retained.column_with_name(ROW_ADDR).is_some());
        assert_eq!(dropped.fields().len() + 1, retained.fields().len());
    }

    #[tokio::test]
    async fn a_blob_column_already_in_logical_layout_needs_no_rewrite() {
        let dir = TempStrDir::default();
        let dataset = blob_v2_dataset(&dir, 2).await;

        let DataType::Struct(logical_fields) = BLOB_V2_LOGICAL_TYPE.clone() else {
            panic!("the logical blob v2 type should be a struct");
        };
        assert_eq!(
            BlobV2Layout::classify(&logical_fields),
            Some(BlobV2Layout::Logical),
            "this test only pins the fix if its input column really is in logical layout"
        );

        // Same column name as the dataset's blob column, so a name-only check would
        // demand a row address column that this schema deliberately does not carry.
        let input_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt32, false),
            Field::new("blob", DataType::Struct(logical_fields.clone()), true),
        ]));
        assert!(input_schema.column_with_name(ROW_ADDR).is_none());

        let plan = BlobRewritePlan::try_new(dataset.clone(), input_schema.as_ref(), false).unwrap();
        assert_eq!(plan.output_schema(), &input_schema);

        let blobs = StructArray::new(
            logical_fields,
            vec![
                Arc::new(LargeBinaryArray::from_iter_values([
                    b"zero".as_slice(),
                    b"one".as_slice(),
                ])),
                Arc::new(StringArray::new_null(2)),
                Arc::new(UInt64Array::new_null(2)),
                Arc::new(UInt64Array::new_null(2)),
            ],
            None,
        );
        let input = RecordBatch::try_new(
            input_schema.clone(),
            vec![
                Arc::new(UInt32Array::from_iter_values(0..2)),
                Arc::new(blobs),
            ],
        )
        .unwrap();

        let output = plan.transform(input.clone()).await.unwrap();
        assert_eq!(output, input);
    }

    #[tokio::test]
    async fn the_managed_decision_follows_the_dataset() {
        let dir = TempStrDir::default();
        let dataset = blob_v2_dataset(&dir, 1).await;
        let input_schema = scan_with_row_addresses(&dataset).await[0].schema();

        let plan = BlobRewritePlan::try_new(dataset.clone(), input_schema.as_ref(), false).unwrap();
        assert_eq!(
            BlobV2Layout::classify(&blob_output_fields(plan.output_schema(), "blob")),
            Some(BlobV2Layout::Logical)
        );
    }

    /// A Blob v2 array of `rows` payloads, each 16 bytes of the row index.
    fn blob_values(rows: usize) -> ArrayRef {
        let mut blobs = BlobArrayBuilder::new(rows);
        for row in 0..rows {
            blobs.push_bytes(vec![row as u8; 16]).unwrap();
        }
        blobs.finish().unwrap()
    }

    /// Write a single-column dataset at V2_2, so its blob payloads are stored
    /// out of line and the column scans back in descriptor layout.
    async fn nested_dataset(dir: &TempStrDir, field: Field, array: ArrayRef) -> Arc<Dataset> {
        let schema = Arc::new(Schema::new(vec![field]));
        let batch = RecordBatch::try_new(schema.clone(), vec![array]).unwrap();
        Arc::new(
            Dataset::write(
                RecordBatchIterator::new([Ok(batch)], schema),
                dir,
                Some(WriteParams {
                    data_storage_version: Some(LanceFileVersion::V2_2),
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        )
    }

    /// The blob struct reached through a list column, by descending `path` from the
    /// named column through list items and struct children.
    fn nested_blob_fields(schema: &ArrowSchema, column: &str, path: &[&str]) -> Fields {
        let mut data_type = schema.field_with_name(column).unwrap().data_type().clone();
        for step in path {
            data_type = match (&data_type, *step) {
                (DataType::List(item), "item") | (DataType::LargeList(item), "item") => {
                    item.data_type().clone()
                }
                (DataType::Struct(children), name) => children
                    .find(name)
                    .map(|(_, child)| child.data_type().clone())
                    .unwrap_or_else(|| panic!("struct has no child '{name}'")),
                (other, step) => panic!("cannot descend '{step}' into {other:?}"),
            };
        }
        let DataType::Struct(fields) = data_type else {
            panic!("the path {path:?} should end at the blob struct, got {data_type:?}");
        };
        fields
    }

    /// Assert that a nested descriptor column is found: without row addresses the
    /// plan is refused, and with them the blob struct is rewritten to logical layout.
    ///
    /// Returns the batches scanned with row addresses, so the caller can transform them.
    async fn assert_nested_blob_is_rewritten(
        dataset: &Arc<Dataset>,
        column: &str,
        path: &[&str],
    ) -> Vec<RecordBatch> {
        // (a) The nested descriptor is found, so row addresses are now required.
        let without = scan_without_row_addresses(dataset).await;
        let error = BlobRewritePlan::try_new(dataset.clone(), without.schema().as_ref(), false)
            .unwrap_err();
        assert!(
            matches!(error, Error::InvalidInput { .. }),
            "unexpected error variant: {error:?}"
        );
        assert!(error.to_string().contains(ROW_ADDR), "{error}");

        // (b) With row addresses the plan builds, and the nested blob struct changes
        // from descriptor layout to logical layout.
        let batches = scan_with_row_addresses(dataset).await;
        let input_schema = batches[0].schema();
        assert_eq!(
            BlobV2Layout::classify(&nested_blob_fields(&input_schema, column, path)),
            Some(BlobV2Layout::Descriptor),
            "this test only pins the fix if the column really arrives in descriptor layout"
        );

        let plan = BlobRewritePlan::try_new(dataset.clone(), input_schema.as_ref(), false).unwrap();
        let output_schema = plan.output_schema().clone();
        assert_ne!(
            output_schema.field_with_name(column).unwrap().data_type(),
            input_schema.field_with_name(column).unwrap().data_type(),
            "a rewritten column must not keep its descriptor type"
        );
        assert_eq!(
            BlobV2Layout::classify(&nested_blob_fields(&output_schema, column, path)),
            Some(BlobV2Layout::Logical)
        );
        batches
    }

    /// Every blob payload reachable under a list column, in row order.
    fn list_blob_payloads<O: arrow_array::OffsetSizeTrait>(
        batch: &RecordBatch,
        column: &str,
        struct_child: Option<&str>,
    ) -> Vec<Vec<u8>> {
        let list = batch
            .column_by_name(column)
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::GenericListArray<O>>()
            .unwrap();
        let values = list.values();
        let blobs = match struct_child {
            None => values.clone(),
            Some(name) => values
                .as_any()
                .downcast_ref::<StructArray>()
                .unwrap()
                .column_by_name(name)
                .unwrap()
                .clone(),
        };
        let blobs = blobs.as_any().downcast_ref::<StructArray>().unwrap();
        let data = blobs
            .column_by_name("data")
            .unwrap()
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .unwrap();
        (0..data.len())
            .map(|row| data.value(row).to_vec())
            .collect()
    }

    #[tokio::test]
    async fn a_list_of_descriptor_blobs_is_rewritten_not_passed_through() {
        let dir = TempStrDir::default();
        let item = blob_field("item", true);
        let values = blob_values(2);
        let offsets = OffsetBuffer::new(ScalarBuffer::from(vec![0i32, 1, 2]));
        let list: ArrayRef =
            Arc::new(ListArray::try_new(Arc::new(item.clone()), offsets, values, None).unwrap());
        let dataset = nested_dataset(
            &dir,
            Field::new("blobs", DataType::List(Arc::new(item)), true),
            list,
        )
        .await;

        let batches = assert_nested_blob_is_rewritten(&dataset, "blobs", &["item"]).await;

        // (c) The payloads survive the rewrite and are readable inline.
        let plan =
            BlobRewritePlan::try_new(dataset.clone(), batches[0].schema().as_ref(), false).unwrap();
        let mut payloads = Vec::new();
        for batch in batches {
            let rewritten = plan.transform(batch).await.unwrap();
            assert_eq!(rewritten.schema(), *plan.output_schema());
            payloads.extend(list_blob_payloads::<i32>(&rewritten, "blobs", None));
        }
        assert_eq!(payloads, vec![vec![0u8; 16], vec![1u8; 16]]);
    }

    #[tokio::test]
    async fn a_large_list_of_descriptor_blobs_is_rewritten_not_passed_through() {
        let dir = TempStrDir::default();
        let item = blob_field("item", true);
        let values = blob_values(2);
        let offsets = OffsetBuffer::new(ScalarBuffer::from(vec![0i64, 1, 2]));
        let list: ArrayRef = Arc::new(
            LargeListArray::try_new(Arc::new(item.clone()), offsets, values, None).unwrap(),
        );
        let dataset = nested_dataset(
            &dir,
            Field::new("blobs", DataType::LargeList(Arc::new(item)), true),
            list,
        )
        .await;

        let batches = assert_nested_blob_is_rewritten(&dataset, "blobs", &["item"]).await;

        // (c) The payloads survive the rewrite and are readable inline.
        let plan =
            BlobRewritePlan::try_new(dataset.clone(), batches[0].schema().as_ref(), false).unwrap();
        let mut payloads = Vec::new();
        for batch in batches {
            let rewritten = plan.transform(batch).await.unwrap();
            assert_eq!(rewritten.schema(), *plan.output_schema());
            payloads.extend(list_blob_payloads::<i64>(&rewritten, "blobs", None));
        }
        assert_eq!(payloads, vec![vec![0u8; 16], vec![1u8; 16]]);
    }

    #[tokio::test]
    async fn a_list_of_structs_containing_a_blob_is_rewritten_not_passed_through() {
        let dir = TempStrDir::default();
        let item_fields: Fields = vec![
            Field::new("name", DataType::Utf8, false),
            blob_field("blob", true),
        ]
        .into();
        let values: ArrayRef = Arc::new(
            StructArray::try_new(
                item_fields.clone(),
                vec![
                    Arc::new(StringArray::from_iter_values(["zero", "one"])) as ArrayRef,
                    blob_values(2),
                ],
                None,
            )
            .unwrap(),
        );
        let item = Arc::new(Field::new("item", DataType::Struct(item_fields), true));
        let offsets = OffsetBuffer::new(ScalarBuffer::from(vec![0i32, 1, 2]));
        let list: ArrayRef =
            Arc::new(ListArray::try_new(item.clone(), offsets, values, None).unwrap());
        let dataset =
            nested_dataset(&dir, Field::new("items", DataType::List(item), true), list).await;

        let batches = assert_nested_blob_is_rewritten(&dataset, "items", &["item", "blob"]).await;

        // (c) The payloads survive the rewrite and are readable inline.
        let plan =
            BlobRewritePlan::try_new(dataset.clone(), batches[0].schema().as_ref(), false).unwrap();
        let mut payloads = Vec::new();
        for batch in batches {
            let rewritten = plan.transform(batch).await.unwrap();
            assert_eq!(rewritten.schema(), *plan.output_schema());
            payloads.extend(list_blob_payloads::<i32>(&rewritten, "items", Some("blob")));
        }
        assert_eq!(payloads, vec![vec![0u8; 16], vec![1u8; 16]]);
    }

    /// The managed blob column used by the acceptance tests below. The small
    /// thresholds force the payload out of the data file, so with managed blobs
    /// enabled it becomes a Lance-owned object reached through a descriptor.
    fn managed_blob_column() -> Field {
        blob_field_with_options(
            "blob",
            true,
            BlobFieldOptions {
                inline_size_threshold: Some(4),
                dedicated_size_threshold: Some(NonZeroUsize::new(8).unwrap()),
            },
        )
    }

    /// A one-row managed Blob v2 dataset holding `payload`.
    ///
    /// `base` registers a table base and writes the data file into it, which is
    /// what makes a preserved descriptor name that base explicitly instead of
    /// being relative to whichever dataset later holds it.
    async fn managed_blob_dataset(
        dir: &TempStrDir,
        payload: &[u8],
        base: Option<BasePath>,
    ) -> Arc<Dataset> {
        let schema = Arc::new(managed_blob_schema(vec![
            Field::new("id", DataType::UInt32, false),
            managed_blob_column(),
        ]));
        let mut blobs = BlobArrayBuilder::new(1);
        blobs.push_bytes(payload).unwrap();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from_iter_values(0..1)),
                blobs.finish().unwrap(),
            ],
        )
        .unwrap();
        let base_id = base.as_ref().map(|base| base.id);
        let dataset = Arc::new(
            Dataset::write(
                RecordBatchIterator::new([Ok(batch)], schema),
                dir,
                Some(WriteParams {
                    data_storage_version: Some(LanceFileVersion::V2_2),
                    initial_bases: base.map(|base| vec![base]),
                    target_bases: base_id.map(|id| vec![id]),
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );
        assert!(managed_blobs_enabled(dataset.schema()).unwrap());
        assert!(BlobRewritePlan::can_preserve_managed(&dataset).unwrap());
        let stored = dataset.take_blobs_by_indices(&[0], "blob").await.unwrap();
        assert_eq!(
            stored[0].as_ref().unwrap().kind(),
            BlobKind::Managed,
            "the source payload must already be a managed object, otherwise this \
             test would be about materialization rather than preservation"
        );
        dataset
    }

    /// The preserved descriptor of the single blob row in `batch`.
    ///
    /// Returns `(kind, blob_id, uri, blob_size, inline_len)`. `inline_len` is the
    /// length of the prepared `data` child, which is how the payload would travel
    /// if it had been copied into the output instead of referenced.
    fn preserved_descriptor(batch: &RecordBatch) -> (BlobKind, Option<u32>, String, u64, usize) {
        let blob = batch.column_by_name("blob").unwrap().as_struct();
        assert_eq!(
            BlobV2Layout::classify(blob.fields()),
            Some(BlobV2Layout::Prepared),
            "a preserved managed reference must arrive in prepared layout"
        );
        assert_eq!(blob.len(), 1);
        let kind = BlobKind::try_from(
            blob.column_by_name("kind")
                .unwrap()
                .as_primitive::<arrow_array::types::UInt8Type>()
                .value(0),
        )
        .unwrap();
        let blob_ids = blob
            .column_by_name("blob_id")
            .unwrap()
            .as_primitive::<arrow_array::types::UInt32Type>();
        let data = blob.column_by_name("data").unwrap().as_binary::<i64>();
        (
            kind,
            blob_ids.is_valid(0).then(|| blob_ids.value(0)),
            blob.column_by_name("uri")
                .unwrap()
                .as_string::<i32>()
                .value(0)
                .to_string(),
            blob.column_by_name("blob_size")
                .unwrap()
                .as_primitive::<arrow_array::types::UInt64Type>()
                .value(0),
            if data.is_null(0) {
                0
            } else {
                data.value(0).len()
            },
        )
    }

    /// Transform every row of `source` and write the result to a fresh dataset at
    /// `dir`, registering `bases` there. Returns the transformed batch so the
    /// caller can assert on the descriptor it committed.
    async fn write_preserved_to_destination(
        source: &Arc<Dataset>,
        dir: &TempStrDir,
        bases: Option<Vec<BasePath>>,
    ) -> (RecordBatch, Arc<Dataset>) {
        let batches = scan_with_row_addresses(source).await;
        assert_eq!(batches.len(), 1);
        let plan =
            BlobRewritePlan::try_new(source.clone(), batches[0].schema().as_ref(), false).unwrap();
        let output_schema = plan.output_schema().clone();
        let transformed = plan.transform(batches[0].clone()).await.unwrap();
        assert_eq!(transformed.schema(), output_schema);

        Dataset::write(
            RecordBatchIterator::new([Ok(transformed.clone())], output_schema),
            dir,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                initial_bases: bases,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        (transformed, Arc::new(Dataset::open(dir).await.unwrap()))
    }

    /// Replace every row of the dataset at `dir`, then run its retention with no
    /// retained history, which is what makes the replaced objects removable.
    async fn replace_rows_and_clean(dir: &TempStrDir) {
        let rows = arrow_array::record_batch!(("id", Int32, [0])).unwrap();
        Dataset::write(
            RecordBatchIterator::new([Ok(rows.clone())], rows.schema()),
            dir,
            Some(WriteParams {
                mode: WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let dataset = Dataset::open(dir).await.unwrap();
        let removed = cleanup_old_versions(
            &dataset,
            CleanupPolicy {
                before_version: Some(dataset.version_id()),
                // The replacement commit is seconds old, so only an unfiltered
                // listing reaches the files it superseded.
                delete_unverified: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert!(
            removed.data_files_removed > 0,
            "the replacement did not make any source file removable, so this test \
             would not exercise cleanup at all"
        );
    }

    #[tokio::test]
    async fn a_managed_payload_is_referenced_not_copied_and_reads_back_from_its_base() {
        let source_dir = TempStrDir::default();
        let payload = vec![7u8; 64];
        let base = BasePath::new(
            1,
            source_dir.to_string(),
            Some("payloads".to_string()),
            true,
        );
        let source = managed_blob_dataset(&source_dir, &payload, Some(base.clone())).await;

        let dest_dir = TempStrDir::default();
        let (transformed, dest) =
            write_preserved_to_destination(&source, &dest_dir, Some(vec![base])).await;

        // The output names the source's base and carries no payload bytes: the
        // only thing that crossed into the destination is a reference.
        let (kind, blob_id, uri, size, inline_len) = preserved_descriptor(&transformed);
        assert_eq!(kind, BlobKind::ManagedWithBase);
        assert_eq!(blob_id, Some(1));
        assert_eq!(size, payload.len() as u64);
        assert_eq!(
            inline_len, 0,
            "the payload was copied into the output instead of being referenced"
        );
        assert!(uri.starts_with("_blobs/"), "{uri}");

        // Reading the destination back resolves that reference to the source's
        // object and yields the original bytes.
        let read = dest.take_blobs_by_indices(&[0], "blob").await.unwrap();
        assert_eq!(read[0].as_ref().unwrap().kind(), BlobKind::Managed);
        assert_eq!(
            read[0].as_ref().unwrap().read().await.unwrap().as_ref(),
            payload
        );
    }

    /// A preserved reference is only a reference: the destination is not part of
    /// the source's lineage, so the source's own retention does not know that
    /// anything still needs its objects.
    ///
    /// This pins the current behaviour rather than a desired one. Cleanup of the
    /// source after its rows are replaced removes the referenced object, and the
    /// destination then fails to read the payload it committed.
    #[tokio::test]
    async fn source_cleanup_after_replacement_breaks_the_preserved_reference() {
        let source_dir = TempStrDir::default();
        let payload = vec![7u8; 64];
        let base = BasePath::new(
            1,
            source_dir.to_string(),
            Some("payloads".to_string()),
            true,
        );
        let source = managed_blob_dataset(&source_dir, &payload, Some(base.clone())).await;

        let dest_dir = TempStrDir::default();
        let (transformed, dest) =
            write_preserved_to_destination(&source, &dest_dir, Some(vec![base])).await;
        let (_, _, uri, _, _) = preserved_descriptor(&transformed);
        assert_eq!(
            dest.take_blobs_by_indices(&[0], "blob").await.unwrap()[0]
                .as_ref()
                .unwrap()
                .read()
                .await
                .unwrap()
                .as_ref(),
            payload
        );

        // Drop the open source snapshot before replacing it: keeping it alive
        // would only hide the question this test is asking.
        drop(source);
        replace_rows_and_clean(&source_dir).await;

        let dest = Arc::new(Dataset::open(&dest_dir).await.unwrap());
        let read = dest.take_blobs_by_indices(&[0], "blob").await.unwrap();
        let error = read[0].as_ref().unwrap().read().await.unwrap_err();
        assert!(
            matches!(error, Error::NotFound { .. }),
            "unexpected error variant: {error:?}"
        );
        assert!(error.to_string().contains(&uri), "{error}");
    }

    /// Without a registered base the preserved reference is relative to whichever
    /// dataset holds the descriptor, so a destination elsewhere resolves it
    /// against its own root and finds nothing.
    ///
    /// This pins the current behaviour. The write succeeds, which is why the
    /// registered-base form above is the one that actually survives a readback.
    #[tokio::test]
    async fn a_primary_storage_reference_does_not_resolve_at_another_dataset() {
        let source_dir = TempStrDir::default();
        let payload = vec![7u8; 64];
        let source = managed_blob_dataset(&source_dir, &payload, None).await;

        let dest_dir = TempStrDir::default();
        let (transformed, dest) = write_preserved_to_destination(&source, &dest_dir, None).await;

        let (kind, blob_id, uri, _, inline_len) = preserved_descriptor(&transformed);
        assert_eq!(kind, BlobKind::Managed);
        assert_eq!(blob_id, None, "a plain managed reference names no base");
        assert_eq!(inline_len, 0);

        let read = dest.take_blobs_by_indices(&[0], "blob").await.unwrap();
        let error = read[0].as_ref().unwrap().read().await.unwrap_err();
        assert!(
            matches!(error, Error::NotFound { .. }),
            "unexpected error variant: {error:?}"
        );
        assert!(error.to_string().contains(&uri), "{error}");
        assert!(
            error
                .to_string()
                .contains(dest_dir.as_str().trim_start_matches('/')),
            "the reference resolved somewhere other than the destination root: {error}"
        );
    }
}
