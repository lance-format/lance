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
use arrow_schema::{DataType as ArrowDataType, Schema as ArrowSchema, SchemaRef};
use lance_core::ROW_ADDR;
use lance_core::datatypes::BlobV2Layout;
use lance_file::version::ConcreteFileVersion;
use lance_table::feature_flags::managed_blobs_enabled;

use super::rewrite::{BlobV2BatchRewritePlan, field_contains_blob_v2};
use crate::dataset::Dataset;
use crate::{Error, Result};

/// Whether a column arrives in a layout a Blob v2 rewrite has to act on.
///
/// Only the descriptor layout needs rewriting: it names a payload the new
/// fragment must either reference again or materialize. A column already in
/// logical layout is in its final form, and one that is not a Blob v2 struct at
/// all — a caller that materialized the column itself, or reused the name for
/// something else — is not this rewrite's business. Nested blobs are reached
/// through the struct children.
fn arrives_as_descriptor(data_type: &ArrowDataType) -> bool {
    let ArrowDataType::Struct(fields) = data_type else {
        return false;
    };
    if BlobV2Layout::classify(fields) == Some(BlobV2Layout::Descriptor) {
        return true;
    }
    fields
        .iter()
        .any(|field| arrives_as_descriptor(field.data_type()))
}

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
        let rewrites_blobs = dataset
            .schema()
            .fields
            .iter()
            .filter(|field| field_contains_blob_v2(field))
            .any(|field| {
                input_schema
                    .column_with_name(&field.name)
                    .is_some_and(|(_, input)| arrives_as_descriptor(input.data_type()))
            });

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

        let preserve_managed = Self::can_preserve_managed(&dataset)?;
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
    use crate::blob::BlobArrayBuilder;
    use crate::blob_field;
    use crate::dataset::WriteParams;
    use arrow_array::{
        LargeBinaryArray, RecordBatchIterator, StringArray, StructArray, UInt32Array, UInt64Array,
    };
    use arrow_schema::{DataType, Field, Fields, Schema};
    use futures::TryStreamExt;
    use lance_core::datatypes::{BLOB_V2_LOGICAL_TYPE, BlobV2Layout};
    use lance_core::utils::tempfile::TempStrDir;
    use lance_file::version::LanceFileVersion;

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
}
