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
use lance_core::datatypes::Field as LanceField;
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
/// properties of the dataset, which is why the choice is derived rather than
/// offered. It is the safe default, not the only correct policy: a caller could
/// legitimately prefer materializing a payload it could have preserved, at the
/// cost of copying it, and that knob is simply not part of this surface.
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
/// holds a reference to an object under the *source* dataset's blob storage; the
/// plan does not retain that object.
///
/// The supported destination is a fragment rewrite of the same dataset, writing
/// either to its primary storage or to the registered base named by
/// [`Self::with_target_base_id`]. The caller must keep the source snapshot alive
/// until the new references are committed, as [`preserve_managed_descriptors`]
/// requires; after that, [`cleanup_old_versions`] reads the committed fragments
/// and keeps the payload objects they still reference, so replacing a fragment
/// does not strand its payload.
///
/// A destination in *another* dataset is not supported. The reference names the
/// source files' own base, which is no base at all for primary storage, so it
/// resolves against whichever dataset holds the descriptor; registering a base at
/// the destination does not convert one, and cleanup of the source then breaks it.
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
    ///
    /// # Legacy (v1) blob columns must be scanned as binary
    ///
    /// A legacy blob column read with the default [`BlobHandling`] arrives as a
    /// position/size description of bytes that live elsewhere in the source file,
    /// which is not data a new fragment can republish. This plan has no way to
    /// turn such a description back into its payload — only the scan can, and the
    /// scan belongs to the caller — so a schema carrying one is refused here
    /// rather than passed through as if it were final. Scan the column with
    /// [`BlobHandling::AllBinary`] instead; the materialized binary column is
    /// accepted and passes through unchanged.
    ///
    /// [`BlobHandling`]: lance_core::datatypes::BlobHandling
    /// [`BlobHandling::AllBinary`]: lance_core::datatypes::BlobHandling::AllBinary
    pub fn try_new(
        dataset: Arc<Dataset>,
        input_schema: &ArrowSchema,
        retain_row_addresses: bool,
    ) -> Result<Self> {
        let input_schema = Arc::new(input_schema.clone());
        Self::reject_legacy_blob_descriptions(&dataset, &input_schema)?;
        let preserve_managed = Self::can_preserve_managed(&dataset)?;

        // Only the plan itself knows which columns it would touch, and it refuses to
        // be built without a row address column. So ask it on a schema that is the
        // caller's with a row address column added when missing: one schema clone and
        // one plan construction, no I/O.
        //
        // The probe keeps the inner constructor's `Error::internal` for a missing row
        // address column unreachable from this public type; the user-facing refusal is
        // the `invalid_input` below. It cannot change which columns get a Blob plan
        // either, because `_rowaddr` is never a field of a Lance schema, so
        // `rewrites_anything` answers identically for the probe and the real schema.
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

        let Some((_, row_addr_field)) = input_schema.column_with_name(ROW_ADDR) else {
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
        };

        // A row address is read as a `u64` when a descriptor is resolved, so a
        // column of any other type is refused here instead of failing the downcast
        // on the first batch. A nullable column is accepted, because the scanner's
        // row address field is nullable; the values themselves are checked per
        // batch in `transform`, where a null can actually be observed.
        if row_addr_field.data_type() != &ArrowDataType::UInt64 {
            return Err(Error::invalid_input(format!(
                "the {ROW_ADDR} column must be {} to resolve a blob descriptor to its \
                 payload, but this schema declares it as {}",
                ArrowDataType::UInt64,
                row_addr_field.data_type()
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

    /// Plan for output written into the dataset's registered base `base_id`
    /// instead of into primary storage.
    ///
    /// A preserved reference that carries no explicit base is resolved against the
    /// base of the data file holding it, so writing this plan's output to a base
    /// other than the source objects' one would silently repoint every such
    /// reference at the destination base. Naming the destination here is what makes
    /// those references explicit instead.
    ///
    /// The caller must write the output only to this base, through the same base
    /// registry. A plan describes one destination: use one plan per base rather
    /// than one plan for a writer that spreads rows over several.
    ///
    /// `base_id` must already be registered on the plan's dataset. Preserving rows
    /// that live in primary storage into another base additionally needs the
    /// dataset's own URI to be registered as a base, so that the reference can name
    /// where the payload actually is; without it [`Self::transform`] fails rather
    /// than publishing a reference that resolves elsewhere.
    pub fn with_target_base_id(mut self, base_id: u32) -> Result<Self> {
        if !self.dataset.manifest.base_paths.contains_key(&base_id) {
            return Err(Error::invalid_input(format!(
                "base id {base_id} is not registered on dataset '{}'; a rewrite \
                 destination must be one of its registered bases: {:?}",
                self.dataset.uri(),
                self.dataset
                    .manifest
                    .base_paths
                    .keys()
                    .copied()
                    .collect::<Vec<_>>()
            )));
        }
        self.rewrite = match self.rewrite {
            Rewrite::BlobV2(inner) => Rewrite::BlobV2(inner.with_target_base_id(Some(base_id))),
            // Nothing is preserved, so no descriptor can name a base.
            passthrough => passthrough,
        };
        Ok(self)
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
    /// silently misread. The match covers schema metadata as well as fields, because
    /// the output carries the metadata of the schema the plan was built from. See the
    /// note on this type about the rows' origin, which this check cannot verify.
    ///
    /// When the plan rewrites a Blob v2 column, every row must carry its source
    /// row address: a batch with a null row address is rejected rather than
    /// resolved, because a null names no row to resolve a descriptor against.
    ///
    /// Batches may be transformed concurrently. Depending on the rewrite chosen,
    /// this may read payload bytes from storage, so it is not a cheap in-memory
    /// projection.
    pub async fn transform(&self, batch: RecordBatch) -> Result<RecordBatch> {
        if batch.schema().as_ref() != self.input_schema.as_ref() {
            return Err(Error::invalid_input(format!(
                "this batch does not have the schema its plan was built for (fields and \
                 schema metadata must both match); expected columns {:?} but got {:?}",
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
            Rewrite::BlobV2(inner) => {
                Self::reject_null_row_addresses(&batch)?;
                inner.transform_batch(&self.dataset, batch).await
            }
        }
    }

    /// Refuse a batch whose row address column has a null value.
    ///
    /// A descriptor is resolved by row address, and a null address names no row.
    /// The resolver reads the values buffer, where a null slot still holds some
    /// `u64`, so letting one through would resolve another row's payload instead
    /// of failing. Checked per batch rather than at construction because a
    /// nullable column is a legitimate input as long as its values are present.
    fn reject_null_row_addresses(batch: &RecordBatch) -> Result<()> {
        let Some(row_addrs) = batch.column_by_name(ROW_ADDR) else {
            return Ok(());
        };
        let null_count = row_addrs.null_count();
        if null_count > 0 {
            return Err(Error::invalid_input(format!(
                "the {ROW_ADDR} column of this batch has {null_count} null value(s) out of \
                 {}, and a null row address names no row to resolve a blob descriptor \
                 against; supply the address of the source row for every row being \
                 rewritten, or drop the rows that have none before transforming",
                row_addrs.len()
            )));
        }
        Ok(())
    }

    /// Refuse an input schema that projects a legacy (v1) blob column as its
    /// position/size description.
    ///
    /// Whether a column *is* a legacy blob is a property of the dataset, so it is
    /// read from the dataset schema rather than guessed from the Arrow type: an
    /// ordinary struct of the same shape is data and passes through. The Arrow
    /// type is then what distinguishes the two projections of such a field, a
    /// description struct from the materialized binary this plan accepts.
    ///
    /// Only projected fields are examined. A legacy blob column the caller did not
    /// select cannot reach the output, and the rows written for it come from the
    /// dataset's own schema rather than from this batch.
    fn reject_legacy_blob_descriptions(
        dataset: &Arc<Dataset>,
        input_schema: &ArrowSchema,
    ) -> Result<()> {
        fn find(
            dataset_field: &LanceField,
            input_field: &ArrowField,
            path: &mut Vec<String>,
        ) -> Option<String> {
            path.push(input_field.name().clone());
            if dataset_field.is_blob()
                && !dataset_field.is_blob_v2()
                && !matches!(
                    input_field.data_type(),
                    ArrowDataType::Binary | ArrowDataType::LargeBinary | ArrowDataType::BinaryView
                )
            {
                return Some(path.join("."));
            }
            // A blob leaf has no children worth descending into, so only the
            // nesting that can *contain* one is walked.
            let found = match input_field.data_type() {
                ArrowDataType::Struct(children) => children
                    .iter()
                    .filter_map(|child| {
                        let dataset_child = dataset_field.child(child.name())?;
                        find(dataset_child, child, path)
                    })
                    .next(),
                // A map's entries struct is its only child in both
                // representations, so it is walked like a list's item rather than
                // looked up by name, whose Arrow and Lance spellings may differ.
                ArrowDataType::List(child)
                | ArrowDataType::LargeList(child)
                | ArrowDataType::ListView(child)
                | ArrowDataType::LargeListView(child)
                | ArrowDataType::FixedSizeList(child, _)
                | ArrowDataType::Map(child, _) => dataset_field
                    .children
                    .first()
                    .and_then(|dataset_child| find(dataset_child, child, path)),
                _ => None,
            };
            path.pop();
            found
        }

        let schema = dataset.schema();
        let offender = input_schema.fields().iter().find_map(|input_field| {
            let dataset_field = schema.field(input_field.name())?;
            find(dataset_field, input_field, &mut Vec::new())
        });
        if let Some(path) = offender {
            return Err(Error::invalid_input(format!(
                "column '{path}' is a legacy blob column of dataset '{}' projected as its \
                 position/size description, which describes bytes in the source file rather \
                 than carrying them, so rewriting it would publish the description in place \
                 of the payload; scan it with BlobHandling::AllBinary so the payload is \
                 materialized as binary",
                dataset.uri()
            )));
        }
        Ok(())
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
    use crate::dataset::optimize::execution::rechunk_row_lineage;
    use crate::dataset::transaction::{Operation, RewriteGroup, Transaction};
    use crate::dataset::{CommitBuilder, InsertBuilder, WriteMode, WriteParams};
    use arrow_array::Array;
    use arrow_array::cast::AsArray;
    use arrow_array::{
        ArrayRef, Int32Array, LargeBinaryArray, LargeListArray, ListArray, RecordBatchIterator,
        StringArray, StructArray, UInt32Array, UInt64Array,
    };
    use arrow_buffer::{OffsetBuffer, ScalarBuffer};
    use arrow_schema::{DataType, Field, Fields, Schema};
    use futures::TryStreamExt;
    use lance_core::datatypes::{BLOB_V2_LOGICAL_TYPE, BlobKind, BlobV2Layout};
    use lance_core::utils::tempfile::TempStrDir;
    use lance_file::version::LanceFileVersion;
    use lance_table::format::BasePath;
    use lance_table::format::Fragment;
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

    /// A legacy (v1) blob field: the blob metadata key on a binary column, with no
    /// Blob v2 extension name. Legacy blobs are only writable below file version
    /// 2.2, which is why these fixtures write V2_1.
    fn legacy_blob_field(name: &str) -> Field {
        Field::new(name, DataType::LargeBinary, true)
            .with_metadata([(lance_arrow::BLOB_META_KEY.to_string(), "true".to_string())].into())
    }

    const LEGACY_PAYLOADS: [&[u8]; 3] = [b"abc", b"defgh", b"ijklmnop"];

    fn legacy_blob_values() -> ArrayRef {
        Arc::new(LargeBinaryArray::from_iter_values(LEGACY_PAYLOADS))
    }

    async fn write_legacy_at(
        dir: &TempStrDir,
        schema: Arc<Schema>,
        blobs: ArrayRef,
        version: LanceFileVersion,
    ) -> Arc<Dataset> {
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from_iter_values(
                    0..LEGACY_PAYLOADS.len() as u32,
                )),
                blobs,
            ],
        )
        .unwrap();
        Arc::new(
            Dataset::write(
                RecordBatchIterator::new([Ok(batch)], schema),
                dir,
                Some(WriteParams {
                    data_storage_version: Some(version),
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        )
    }

    /// Legacy blobs are only writable below file version 2.2.
    async fn write_legacy(dir: &TempStrDir, schema: Arc<Schema>, blobs: ArrayRef) -> Arc<Dataset> {
        write_legacy_at(dir, schema, blobs, LanceFileVersion::V2_1).await
    }

    async fn legacy_blob_dataset(dir: &TempStrDir) -> Arc<Dataset> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt32, false),
            legacy_blob_field("blob"),
        ]));
        write_legacy(dir, schema, legacy_blob_values()).await
    }

    /// The same payloads behind a legacy blob field nested in a plain struct.
    async fn nested_legacy_blob_dataset(dir: &TempStrDir) -> Arc<Dataset> {
        let inner = legacy_blob_field("blob");
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt32, false),
            Field::new(
                "wrapper",
                DataType::Struct(vec![Arc::new(inner.clone())].into()),
                true,
            ),
        ]));
        let nested: ArrayRef = Arc::new(StructArray::new(
            vec![Arc::new(inner)].into(),
            vec![legacy_blob_values()],
            None,
        ));
        write_legacy(dir, schema, nested).await
    }

    /// A map whose entry values are `value_field`, with one entry per row.
    fn map_of(value_field: Field, values: ArrayRef) -> (Field, ArrayRef) {
        let entry_fields =
            Fields::from(vec![Field::new("key", DataType::Utf8, false), value_field]);
        let keys: ArrayRef = Arc::new(StringArray::from_iter_values(
            (0..values.len()).map(|row| format!("k{row}")),
        ));
        let entries = StructArray::new(entry_fields.clone(), vec![keys, values], None);
        let entries_field = Arc::new(Field::new("entries", DataType::Struct(entry_fields), false));
        let maps = arrow_array::MapArray::try_new(
            entries_field.clone(),
            OffsetBuffer::from_lengths(std::iter::repeat_n(1, entries.len())),
            entries,
            None,
            false,
        )
        .unwrap();
        (
            Field::new("props", DataType::Map(entries_field, false), true),
            Arc::new(maps),
        )
    }

    /// A dataset whose map entry value is marked as a legacy blob.
    ///
    /// No file version can hold this on disk: map encoding starts at 2.2, and at
    /// 2.2 a legacy blob marker is rejected in favour of the Blob v2 extension,
    /// while the versions that accept the marker have no map encoding. So the
    /// marker is applied to the opened dataset's in-memory schema, which is the
    /// schema the guard reads, over a map the writer does accept.
    async fn map_legacy_blob_dataset(dir: &TempStrDir) -> Arc<Dataset> {
        let (map_field, maps) = map_of(
            Field::new("value", DataType::LargeBinary, true),
            legacy_blob_values(),
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt32, false),
            map_field,
        ]));
        let dataset = write_legacy_at(dir, schema, maps, LanceFileVersion::V2_2).await;

        let mut manifest = dataset.manifest.as_ref().clone();
        let entry_value = &mut manifest.schema.fields[1].children[0].children[1];
        assert_eq!(entry_value.name, "value");
        entry_value
            .metadata
            .insert(lance_arrow::BLOB_META_KEY.to_string(), "true".to_string());
        assert!(entry_value.is_blob() && !entry_value.is_blob_v2());
        let mut dataset = dataset.as_ref().clone();
        dataset.manifest = Arc::new(manifest);
        Arc::new(dataset)
    }

    async fn scan_all_binary(dataset: &Dataset) -> RecordBatch {
        let mut scan = dataset.scan();
        scan.blob_handling(lance_core::datatypes::BlobHandling::AllBinary);
        scan.try_into_batch().await.unwrap()
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
    async fn a_legacy_blob_projected_as_a_description_is_refused_with_the_binary_scan_to_use() {
        let dir = TempStrDir::default();
        let dataset = legacy_blob_dataset(&dir).await;
        let input = scan_without_row_addresses(&dataset).await;
        // The default handling yields the description, which is what makes this
        // schema unusable as rewrite input.
        assert!(matches!(
            input.schema().field_with_name("blob").unwrap().data_type(),
            DataType::Struct(_)
        ));

        let error =
            BlobRewritePlan::try_new(dataset.clone(), input.schema().as_ref(), false).unwrap_err();
        assert!(
            matches!(error, Error::InvalidInput { .. }),
            "unexpected error variant: {error:?}"
        );
        assert!(
            error.to_string().contains("BlobHandling::AllBinary"),
            "{error}"
        );
        assert!(error.to_string().contains("blob"), "{error}");
    }

    #[tokio::test]
    async fn a_legacy_blob_scanned_as_binary_passes_its_payload_through_to_a_written_dataset() {
        let dir = TempStrDir::default();
        let dataset = legacy_blob_dataset(&dir).await;
        let input = scan_all_binary(&dataset).await;
        assert_eq!(
            input.schema().field_with_name("blob").unwrap().data_type(),
            &DataType::LargeBinary
        );

        let plan =
            BlobRewritePlan::try_new(dataset.clone(), input.schema().as_ref(), false).unwrap();
        let output_schema = plan.output_schema().clone();
        assert_eq!(&output_schema, &input.schema());
        let transformed = plan.transform(input.clone()).await.unwrap();
        assert_eq!(transformed, input);

        let destination = TempStrDir::default();
        Dataset::write(
            RecordBatchIterator::new([Ok(transformed)], output_schema),
            &destination,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_1),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let written = Dataset::open(&destination).await.unwrap();
        // The materialized column keeps its blob marker, so the destination is a
        // legacy blob dataset too and the payload is read back as binary.
        let read_back = scan_all_binary(&written).await;
        let blobs = read_back
            .column_by_name("blob")
            .unwrap()
            .as_binary::<i64>()
            .clone();
        let payloads = (0..blobs.len())
            .map(|row| blobs.value(row))
            .collect::<Vec<_>>();
        assert_eq!(payloads, LEGACY_PAYLOADS.to_vec());
    }

    #[tokio::test]
    async fn a_legacy_blob_nested_in_a_struct_is_refused_when_projected_as_a_description() {
        let dir = TempStrDir::default();
        let dataset = nested_legacy_blob_dataset(&dir).await;
        let input = scan_without_row_addresses(&dataset).await;

        let error =
            BlobRewritePlan::try_new(dataset.clone(), input.schema().as_ref(), false).unwrap_err();
        assert!(
            matches!(error, Error::InvalidInput { .. }),
            "unexpected error variant: {error:?}"
        );
        assert!(error.to_string().contains("wrapper.blob"), "{error}");
        assert!(
            error.to_string().contains("BlobHandling::AllBinary"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn a_legacy_blob_under_a_map_is_refused_when_projected_as_a_description() {
        let dir = TempStrDir::default();
        let dataset = map_legacy_blob_dataset(&dir).await;

        // The input schema is the projection the dataset's own blob handling
        // produces for that field tree. It is built from the schema rather than
        // scanned because the patched field no longer matches the file on disk.
        let projected = lance_core::datatypes::BlobHandling::BlobsDescriptions
            .unload_if_needed(dataset.schema().fields[1].clone());
        let input_schema = ArrowSchema::new(vec![
            ArrowField::new("id", DataType::UInt32, false),
            ArrowField::from(&projected),
        ]);
        let DataType::Map(entries, _) = input_schema.field_with_name("props").unwrap().data_type()
        else {
            panic!("'props' should be a map");
        };
        let DataType::Struct(entry_fields) = entries.data_type() else {
            panic!("map entries should be a struct");
        };
        assert!(
            matches!(entry_fields[1].data_type(), DataType::Struct(_)),
            "the entry value should be projected as a description struct"
        );

        let error = BlobRewritePlan::try_new(dataset, &input_schema, false).unwrap_err();
        assert!(
            matches!(error, Error::InvalidInput { .. }),
            "unexpected error variant: {error:?}"
        );
        assert!(error.to_string().contains("props"), "{error}");
        assert!(error.to_string().contains("value"), "{error}");
        assert!(
            error.to_string().contains("BlobHandling::AllBinary"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn an_ordinary_map_with_no_blob_metadata_is_accepted() {
        let dir = TempStrDir::default();
        let (map_field, maps) = map_of(
            Field::new("value", DataType::LargeBinary, true),
            legacy_blob_values(),
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt32, false),
            map_field,
        ]));
        let dataset = write_legacy_at(&dir, schema, maps, LanceFileVersion::V2_2).await;

        let input = scan_without_row_addresses(&dataset).await;
        let plan =
            BlobRewritePlan::try_new(dataset.clone(), input.schema().as_ref(), false).unwrap();
        assert_eq!(plan.transform(input.clone()).await.unwrap(), input);
    }

    /// A column is a legacy blob because the dataset says so, not because of its
    /// shape, so an ordinary struct of position and size is ordinary data.
    #[tokio::test]
    async fn a_plain_struct_shaped_like_a_blob_description_is_accepted() {
        let dir = TempStrDir::default();
        let description_like = Fields::from(vec![
            Field::new("position", DataType::UInt64, true),
            Field::new("size", DataType::UInt64, true),
        ]);
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt32, false),
            Field::new("blob", DataType::Struct(description_like.clone()), true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from_iter_values(0..2)),
                Arc::new(StructArray::new(
                    description_like,
                    vec![
                        Arc::new(UInt64Array::from_iter_values(0..2)),
                        Arc::new(UInt64Array::from_iter_values(10..12)),
                    ],
                    None,
                )),
            ],
        )
        .unwrap();
        let dataset = Arc::new(
            Dataset::write(
                RecordBatchIterator::new([Ok(batch)], schema),
                &dir,
                Some(WriteParams {
                    data_storage_version: Some(LanceFileVersion::V2_2),
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );

        let input = scan_without_row_addresses(&dataset).await;
        let plan =
            BlobRewritePlan::try_new(dataset.clone(), input.schema().as_ref(), false).unwrap();
        assert_eq!(plan.transform(input.clone()).await.unwrap(), input);
    }

    #[tokio::test]
    async fn a_row_address_column_of_the_wrong_type_is_refused_when_the_plan_is_built() {
        let dir = TempStrDir::default();
        let dataset = blob_v2_dataset(&dir, 2).await;
        let scanned = scan_with_row_addresses(&dataset).await[0].schema();

        let retyped = Arc::new(ArrowSchema::new_with_metadata(
            scanned
                .fields()
                .iter()
                .map(|field| {
                    if field.name() == ROW_ADDR {
                        Arc::new(ArrowField::new(ROW_ADDR, DataType::Int64, false))
                    } else {
                        field.clone()
                    }
                })
                .collect::<Vec<_>>(),
            scanned.metadata().clone(),
        ));

        let error = BlobRewritePlan::try_new(dataset.clone(), retyped.as_ref(), false).unwrap_err();
        assert!(
            matches!(error, Error::InvalidInput { .. }),
            "unexpected error variant: {error:?}"
        );
        assert!(error.to_string().contains(ROW_ADDR), "{error}");
        assert!(error.to_string().contains("Int64"), "{error}");

        // The scanner's own column is still accepted.
        BlobRewritePlan::try_new(dataset, scanned.as_ref(), false).unwrap();
    }

    /// Replace the row address column of `batch` with `addresses`, keeping the
    /// nullable row address field the scanner declares.
    fn with_row_addresses(batch: &RecordBatch, addresses: UInt64Array) -> RecordBatch {
        let (index, _) = batch.schema().column_with_name(ROW_ADDR).unwrap();
        let mut fields = batch.schema().fields().to_vec();
        fields[index] = Arc::new(ArrowField::new(ROW_ADDR, DataType::UInt64, true));
        let mut columns = batch.columns().to_vec();
        columns[index] = Arc::new(addresses);
        RecordBatch::try_new(
            Arc::new(ArrowSchema::new_with_metadata(
                fields,
                batch.schema().metadata().clone(),
            )),
            columns,
        )
        .unwrap()
    }

    #[tokio::test]
    async fn a_nullable_row_address_column_of_present_addresses_is_transformed() {
        let dir = TempStrDir::default();
        let dataset = blob_v2_dataset(&dir, 1).await;
        let scanned = scan_with_row_addresses(&dataset).await[0].clone();
        let addresses = scanned
            .column_by_name(ROW_ADDR)
            .unwrap()
            .as_primitive::<arrow_array::types::UInt64Type>()
            .clone();
        let nullable = with_row_addresses(
            &scanned,
            UInt64Array::from_iter(addresses.values().iter().copied().map(Some)),
        );

        let plan =
            BlobRewritePlan::try_new(dataset.clone(), nullable.schema().as_ref(), false).unwrap();
        let output = plan.transform(nullable).await.unwrap();
        assert_eq!(&output.schema(), plan.output_schema());
        let blob = output.column_by_name("blob").unwrap().as_struct();
        let data = blob.column_by_name("data").unwrap().as_binary::<i64>();
        assert_eq!(data.value(0), vec![0u8; 16]);
    }

    #[tokio::test]
    async fn a_null_row_address_is_refused_instead_of_resolved_against_its_slot() {
        let dir = TempStrDir::default();
        let dataset = blob_v2_dataset(&dir, 1).await;
        let scanned = scan_with_row_addresses(&dataset).await[0].clone();
        // A null whose backing slot holds a real address: resolving the buffer
        // instead of the validity would silently return that row's payload.
        let addresses = scanned
            .column_by_name(ROW_ADDR)
            .unwrap()
            .as_primitive::<arrow_array::types::UInt64Type>()
            .clone();
        let with_null = with_row_addresses(
            &scanned,
            UInt64Array::from_iter(addresses.values().iter().map(|_| None::<u64>)),
        );

        let plan =
            BlobRewritePlan::try_new(dataset.clone(), with_null.schema().as_ref(), false).unwrap();
        let error = plan.transform(with_null).await.unwrap_err();
        assert!(
            matches!(error, Error::InvalidInput { .. }),
            "unexpected error variant: {error:?}"
        );
        assert!(error.to_string().contains(ROW_ADDR), "{error}");
        assert!(error.to_string().contains("1 null value(s)"), "{error}");
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

    /// The backing object a blob row resolves to: its path, byte range and kind.
    ///
    /// A preserved reference and a fresh copy of the same payload both read back
    /// equal bytes, so only the object identity distinguishes them.
    async fn blob_object_identity(
        dataset: &Arc<Dataset>,
        column: &str,
    ) -> (String, u64, u64, BlobKind) {
        let blobs = dataset.take_blobs_by_indices(&[0], column).await.unwrap();
        let blob = blobs[0].as_ref().unwrap();
        (
            blob.data_path().to_string(),
            blob.position(),
            blob.size(),
            blob.kind(),
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

    /// The case this plan is for: the rewritten rows replace their own fragments
    /// in the dataset they came from. Cleanup then sees the preserved references
    /// in the latest manifest's fragments, so it keeps their payload objects even
    /// though the fragments that first wrote them are gone.
    #[tokio::test]
    async fn a_same_dataset_rewrite_keeps_its_preserved_payload_through_cleanup() {
        let dir = TempStrDir::default();
        let payload = vec![7u8; 64];
        let dataset = managed_blob_dataset(&dir, &payload, None).await;

        let source_object = blob_object_identity(&dataset, "blob").await;
        let old_fragments = dataset.fragments().to_vec();
        let batches = scan_with_row_addresses(&dataset).await;
        assert_eq!(batches.len(), 1);
        let plan =
            BlobRewritePlan::try_new(dataset.clone(), batches[0].schema().as_ref(), false).unwrap();
        let transformed = plan.transform(batches[0].clone()).await.unwrap();
        assert_eq!(transformed.schema(), *plan.output_schema());

        // Write the rewritten rows as new fragments of this same dataset and swap
        // them in, the way compaction does.
        let written = InsertBuilder::new(dataset.clone())
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            })
            .execute_uncommitted(vec![transformed])
            .await
            .unwrap();
        let Operation::Append {
            fragments: mut new_fragments,
        } = written.operation
        else {
            unreachable!("an uncommitted insert always carries Append");
        };
        if dataset.manifest.uses_stable_row_ids() {
            rechunk_row_lineage(dataset.as_ref(), &mut new_fragments, &old_fragments)
                .await
                .unwrap();
        }
        let committed = CommitBuilder::new(dataset.clone())
            .execute(Transaction::new_from_version(
                dataset.manifest.version,
                Operation::Rewrite {
                    groups: vec![RewriteGroup {
                        old_fragments: old_fragments.clone(),
                        new_fragments,
                    }],
                    rewritten_indices: vec![],
                    frag_reuse_index: None,
                },
            ))
            .await
            .unwrap();

        // Drop the pre-rewrite snapshot: holding it open would protect the old
        // files and hide what cleanup does.
        drop(dataset);
        let removed = cleanup_old_versions(
            &committed,
            CleanupPolicy {
                before_version: Some(committed.version_id()),
                delete_unverified: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert!(
            removed.data_files_removed > 0,
            "the rewrite left every source file in use, so this test would not \
             exercise cleanup at all"
        );
        assert!(
            removed.old_versions > 0,
            "the versions holding the source fragments are still readable, so \
             nothing stops them from keeping the payload alive"
        );

        // The payload object survived, and the committed descriptor is still a
        // reference to it rather than a fresh copy.
        let reopened = Arc::new(Dataset::open(&dir).await.unwrap());
        let read = reopened.take_blobs_by_indices(&[0], "blob").await.unwrap();
        let blob = read[0].as_ref().unwrap();
        assert_eq!(blob.kind(), BlobKind::Managed);
        assert_eq!(blob.read().await.unwrap().as_ref(), payload);
        // Equal bytes alone would also hold for a fresh copy, so pin the object:
        // the committed descriptor still points at the very bytes the source row
        // did, at the same path, offset and length.
        assert_eq!(
            blob_object_identity(&reopened, "blob").await,
            source_object,
            "the rewrite published a different object instead of preserving the \
             source reference"
        );
    }

    /// A managed Blob v2 dataset whose rows live in primary storage, with
    /// `bases` registered but not written to.
    ///
    /// Separate from `managed_blob_dataset`, which writes its data into the base
    /// it registers: here the payload must start in primary storage so that the
    /// destination base differs from the source objects' location.
    async fn primary_managed_dataset_with_bases(
        dir: &TempStrDir,
        payload: &[u8],
        bases: Vec<BasePath>,
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
        let dataset = Arc::new(
            Dataset::write(
                RecordBatchIterator::new([Ok(batch)], schema),
                dir,
                Some(WriteParams {
                    data_storage_version: Some(LanceFileVersion::V2_2),
                    initial_bases: Some(bases),
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );
        assert!(managed_blobs_enabled(dataset.schema()).unwrap());
        let stored = dataset.take_blobs_by_indices(&[0], "blob").await.unwrap();
        assert_eq!(
            stored[0].as_ref().unwrap().kind(),
            BlobKind::Managed,
            "the source payload must be a managed object in primary storage for \
             the destination base to differ from it"
        );
        dataset
    }

    /// Publish `transformed` as new fragments of `dataset`, written into
    /// `target_base`, replacing `old_fragments`.
    async fn publish_rewrite_into_base(
        dataset: &Arc<Dataset>,
        old_fragments: Vec<Fragment>,
        transformed: RecordBatch,
        target_base: Option<u32>,
    ) -> Result<Dataset> {
        let written = InsertBuilder::new(dataset.clone())
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                data_storage_version: Some(LanceFileVersion::V2_2),
                target_bases: target_base.map(|id| vec![id]),
                ..Default::default()
            })
            .execute_uncommitted(vec![transformed])
            .await?;
        let Operation::Append {
            fragments: mut new_fragments,
        } = written.operation
        else {
            unreachable!("an uncommitted insert always carries Append");
        };
        if dataset.manifest.uses_stable_row_ids() {
            rechunk_row_lineage(dataset.as_ref(), &mut new_fragments, &old_fragments).await?;
        }
        CommitBuilder::new(dataset.clone())
            .execute(Transaction::new_from_version(
                dataset.manifest.version,
                Operation::Rewrite {
                    groups: vec![RewriteGroup {
                        old_fragments,
                        new_fragments,
                    }],
                    rewritten_indices: vec![],
                    frag_reuse_index: None,
                },
            ))
            .await
    }

    /// A rewrite whose output goes to a registered base keeps naming the source
    /// objects, because the destination base is declared on the plan.
    ///
    /// `payload` selects the source storage form through the column's size
    /// thresholds (inline 4, dedicated 8): a payload under the dedicated
    /// threshold is packed into a shared sidecar, one at or above it gets its own
    /// object. Both reach `preserve_managed_descriptors` as descriptors whose base
    /// must be made explicit.
    async fn assert_rewrite_into_base_preserves_source_objects(payload: Vec<u8>) {
        let dir = TempStrDir::default();
        // Base 1 is the dataset's own URI: the identity a primary-storage
        // reference needs to stay explicit. Base 2 is where the rewrite writes.
        let identity = BasePath::new(1, dir.to_string(), Some("self".to_string()), true);
        let target_dir = TempStrDir::default();
        let target = BasePath::new(2, target_dir.to_string(), Some("target".to_string()), true);
        let dataset =
            primary_managed_dataset_with_bases(&dir, &payload, vec![identity, target.clone()])
                .await;

        let source_object = blob_object_identity(&dataset, "blob").await;
        let old_fragments = dataset.fragments().to_vec();
        let batches = scan_with_row_addresses(&dataset).await;
        let plan = BlobRewritePlan::try_new(dataset.clone(), batches[0].schema().as_ref(), false)
            .unwrap()
            .with_target_base_id(target.id)
            .unwrap();
        let transformed = plan.transform(batches[0].clone()).await.unwrap();

        // The descriptor names the SOURCE objects' base, not the destination.
        let (kind, blob_id, _, size, inline_len) = preserved_descriptor(&transformed);
        assert_eq!(kind, BlobKind::ManagedWithBase);
        assert_eq!(
            blob_id,
            Some(1),
            "a preserved reference must name the base the payload lives under"
        );
        assert_eq!(size, payload.len() as u64);
        assert_eq!(
            inline_len, 0,
            "the payload was copied instead of referenced"
        );

        let committed =
            publish_rewrite_into_base(&dataset, old_fragments, transformed, Some(target.id))
                .await
                .unwrap();

        let reopened = Arc::new(Dataset::open(committed.uri()).await.unwrap());
        let read = reopened.take_blobs_by_indices(&[0], "blob").await.unwrap();
        assert_eq!(
            read[0].as_ref().unwrap().read().await.unwrap().as_ref(),
            payload
        );
        // The destination resolves to the source's own object, not to a copy of
        // it: same path, offset and length.
        let destination_object = blob_object_identity(&reopened, "blob").await;
        assert_eq!(
            destination_object, source_object,
            "the rewrite published a different object instead of preserving the \
             source reference"
        );
        // And that object lives where the source payload does, not under the base
        // the new fragments were written to.
        assert!(
            destination_object
                .0
                .starts_with(dir.as_str().trim_start_matches('/')),
            "the preserved object should sit under the source base: {}",
            destination_object.0
        );
        assert!(
            !destination_object
                .0
                .starts_with(target_dir.as_str().trim_start_matches('/')),
            "the preserved object resolved under the destination base: {}",
            destination_object.0
        );
    }

    /// T1, source payload in its own managed object.
    #[tokio::test]
    async fn a_rewrite_into_a_registered_base_preserves_a_dedicated_source_object() {
        assert_rewrite_into_base_preserves_source_objects(vec![7u8; 64]).await;
    }

    /// T1, source payload packed into a shared managed object with others.
    #[tokio::test]
    async fn a_rewrite_into_a_registered_base_preserves_a_packed_source_object() {
        assert_rewrite_into_base_preserves_source_objects(vec![9u8; 6]).await;
    }

    /// T2: without a registered base for the dataset's own URI there is no way to
    /// say where a primary-storage payload lives, so the rewrite is refused
    /// instead of publishing a reference that resolves at the destination base.
    #[tokio::test]
    async fn a_rewrite_into_a_base_is_refused_when_the_source_has_no_registered_identity() {
        let dir = TempStrDir::default();
        let payload = vec![7u8; 64];
        let target_dir = TempStrDir::default();
        let target = BasePath::new(2, target_dir.to_string(), Some("target".to_string()), true);
        let dataset =
            primary_managed_dataset_with_bases(&dir, &payload, vec![target.clone()]).await;
        let version_before = dataset.version_id();

        let batches = scan_with_row_addresses(&dataset).await;
        let plan = BlobRewritePlan::try_new(dataset.clone(), batches[0].schema().as_ref(), false)
            .unwrap()
            .with_target_base_id(target.id)
            .unwrap();
        let error = plan.transform(batches[0].clone()).await.unwrap_err();
        assert!(
            matches!(error, Error::InvalidInput { .. }),
            "unexpected error variant: {error:?}"
        );

        // Nothing was published: the refusal happens before any write.
        let reopened = Dataset::open(&dir).await.unwrap();
        assert_eq!(reopened.version_id(), version_before);
        assert_eq!(reopened.fragments().len(), dataset.fragments().len());
    }

    /// T3: a destination the dataset does not know is refused at construction,
    /// where the caller can still act on it.
    #[tokio::test]
    async fn an_unregistered_target_base_is_refused_when_the_plan_is_built() {
        let dir = TempStrDir::default();
        let payload = vec![7u8; 64];
        let dataset = managed_blob_dataset(&dir, &payload, None).await;
        let batches = scan_with_row_addresses(&dataset).await;

        let error = BlobRewritePlan::try_new(dataset.clone(), batches[0].schema().as_ref(), false)
            .unwrap()
            .with_target_base_id(77)
            .unwrap_err();
        assert!(
            matches!(error, Error::InvalidInput { .. }),
            "unexpected error variant: {error:?}"
        );
        assert!(error.to_string().contains("77"), "{error}");
    }

    /// T4: the default destination is primary storage. Writing a plan's output
    /// into a registered base without declaring it leaves every preserved
    /// reference implicit, and an implicit reference is read relative to the file
    /// holding it, so it resolves under the destination base where no payload is.
    #[tokio::test]
    async fn the_default_plan_is_for_primary_destinations_only() {
        let dir = TempStrDir::default();
        let payload = vec![7u8; 64];
        let identity = BasePath::new(1, dir.to_string(), Some("self".to_string()), true);
        let target_dir = TempStrDir::default();
        let target = BasePath::new(2, target_dir.to_string(), Some("target".to_string()), true);
        let dataset =
            primary_managed_dataset_with_bases(&dir, &payload, vec![identity, target.clone()])
                .await;

        let old_fragments = dataset.fragments().to_vec();
        let batches = scan_with_row_addresses(&dataset).await;
        let plan =
            BlobRewritePlan::try_new(dataset.clone(), batches[0].schema().as_ref(), false).unwrap();
        let transformed = plan.transform(batches[0].clone()).await.unwrap();

        // No base is named, which is correct for a primary-storage destination.
        let (kind, blob_id, _, _, _) = preserved_descriptor(&transformed);
        assert_eq!(kind, BlobKind::Managed);
        assert_eq!(blob_id, None);

        let committed =
            publish_rewrite_into_base(&dataset, old_fragments, transformed, Some(target.id))
                .await
                .unwrap();

        // Written into the target base, the implicit reference now resolves
        // there, which is why such a destination must be declared on the plan.
        let reopened = Arc::new(Dataset::open(committed.uri()).await.unwrap());
        let read = reopened.take_blobs_by_indices(&[0], "blob").await.unwrap();
        let error = read[0].as_ref().unwrap().read().await.unwrap_err();
        assert!(
            matches!(error, Error::NotFound { .. }),
            "an undeclared destination base must not silently resolve: {error:?}"
        );
    }

    /// Distinguishable payloads for a struct child, so a child resolved against
    /// the wrong field id is visible in the bytes rather than only in the layout.
    fn named_blob_values(prefix: &str, rows: usize) -> ArrayRef {
        let mut blobs = BlobArrayBuilder::new(rows);
        for row in 0..rows {
            blobs.push_bytes(named_payload(prefix, row)).unwrap();
        }
        blobs.finish().unwrap()
    }

    fn named_payload(prefix: &str, row: usize) -> Vec<u8> {
        format!("payload-{prefix}-{row}").into_bytes()
    }

    /// A dataset with `info: Struct{a: Blob v2, b: Blob v2}`: two children of the
    /// same type, so only the field-to-payload binding tells them apart.
    async fn struct_of_two_blobs_dataset(dir: &TempStrDir, rows: usize) -> Arc<Dataset> {
        let children: Fields = vec![blob_field("a", true), blob_field("b", true)].into();
        let column: ArrayRef = Arc::new(
            StructArray::try_new(
                children.clone(),
                vec![named_blob_values("a", rows), named_blob_values("b", rows)],
                None,
            )
            .unwrap(),
        );
        nested_dataset(
            dir,
            Field::new("info", DataType::Struct(children), true),
            column,
        )
        .await
    }

    /// The same batch with the children of its struct `column` reordered to
    /// `order`, in both the schema and the column's arrays.
    fn with_struct_children_in_order(
        batch: &RecordBatch,
        column: &str,
        order: &[&str],
    ) -> RecordBatch {
        let schema = batch.schema();
        let (column_idx, field) = schema.column_with_name(column).unwrap();
        let DataType::Struct(children) = field.data_type() else {
            panic!("column '{column}' is not a struct");
        };
        let struct_arr = batch.column(column_idx).as_struct();
        let mut new_fields = Vec::with_capacity(order.len());
        let mut new_arrays = Vec::with_capacity(order.len());
        for name in order {
            let (child_idx, child_field) = children.find(name).unwrap();
            new_fields.push(child_field.clone());
            new_arrays.push(struct_arr.column(child_idx).clone());
        }
        let new_fields: Fields = new_fields.into();
        let reordered: ArrayRef = Arc::new(
            StructArray::try_new(new_fields.clone(), new_arrays, struct_arr.nulls().cloned())
                .unwrap(),
        );
        replace_column(
            batch,
            column_idx,
            Arc::new(
                Field::new(column, DataType::Struct(new_fields), field.is_nullable())
                    .with_metadata(field.metadata().clone()),
            ),
            reordered,
        )
    }

    /// The same batch with the struct `column` projected down to `keep`.
    fn with_struct_children_kept(batch: &RecordBatch, column: &str, keep: &[&str]) -> RecordBatch {
        with_struct_children_in_order(batch, column, keep)
    }

    /// The same batch with a struct child added that the dataset does not have.
    fn with_extra_struct_child(
        batch: &RecordBatch,
        column: &str,
        extra: Field,
        values: ArrayRef,
    ) -> RecordBatch {
        let schema = batch.schema();
        let (column_idx, field) = schema.column_with_name(column).unwrap();
        let DataType::Struct(children) = field.data_type() else {
            panic!("column '{column}' is not a struct");
        };
        let struct_arr = batch.column(column_idx).as_struct();
        let mut new_fields = children.to_vec();
        new_fields.push(Arc::new(extra));
        let mut new_arrays = struct_arr.columns().to_vec();
        new_arrays.push(values);
        let new_fields: Fields = new_fields.into();
        let extended: ArrayRef = Arc::new(
            StructArray::try_new(new_fields.clone(), new_arrays, struct_arr.nulls().cloned())
                .unwrap(),
        );
        replace_column(
            batch,
            column_idx,
            Arc::new(
                Field::new(column, DataType::Struct(new_fields), field.is_nullable())
                    .with_metadata(field.metadata().clone()),
            ),
            extended,
        )
    }

    fn replace_column(
        batch: &RecordBatch,
        column_idx: usize,
        field: Arc<Field>,
        array: ArrayRef,
    ) -> RecordBatch {
        let mut fields = batch.schema().fields().to_vec();
        fields[column_idx] = field;
        let mut columns = batch.columns().to_vec();
        columns[column_idx] = array;
        RecordBatch::try_new(
            Arc::new(ArrowSchema::new_with_metadata(
                fields,
                batch.schema().metadata().clone(),
            )),
            columns,
        )
        .unwrap()
    }

    /// Rewrite every batch, publish the result as a `Rewrite` of this same
    /// dataset, and reopen it.
    async fn rewrite_and_republish(dataset: &Arc<Dataset>, batches: Vec<RecordBatch>) -> Dataset {
        let old_fragments = dataset.fragments().to_vec();
        let plan =
            BlobRewritePlan::try_new(dataset.clone(), batches[0].schema().as_ref(), false).unwrap();
        let mut transformed = Vec::with_capacity(batches.len());
        for batch in batches {
            transformed.push(plan.transform(batch).await.unwrap());
        }
        assert_eq!(transformed.len(), 1, "this helper publishes a single batch");
        publish_rewrite_into_base(
            dataset,
            old_fragments,
            transformed.into_iter().next().unwrap(),
            None,
        )
        .await
        .unwrap()
    }

    async fn blob_payloads(dataset: &Arc<Dataset>, column: &str, rows: usize) -> Vec<Vec<u8>> {
        let indices = (0..rows as u64).collect::<Vec<_>>();
        let blobs = dataset
            .take_blobs_by_indices(&indices, column)
            .await
            .unwrap();
        let mut payloads = Vec::with_capacity(rows);
        for blob in blobs {
            payloads.push(blob.unwrap().read().await.unwrap().to_vec());
        }
        payloads
    }

    /// Scanning a struct's blob children in an order other than the dataset's and
    /// rewriting must keep each child's own payload: binding children by position
    /// would publish `b`'s descriptors under `a`'s field id and swap the payloads.
    #[tokio::test]
    async fn reordered_struct_children_keep_their_own_payloads() {
        let dir = TempStrDir::default();
        let rows = 3;
        let dataset = struct_of_two_blobs_dataset(&dir, rows).await;

        let batches = scan_with_row_addresses(&dataset).await;
        assert_eq!(batches.len(), 1);
        let reordered = with_struct_children_in_order(&batches[0], "info", &["b", "a"]);
        assert_eq!(
            BlobV2Layout::classify(&nested_blob_fields(
                reordered.schema().as_ref(),
                "info",
                &["b"]
            )),
            Some(BlobV2Layout::Descriptor),
            "this test only pins the fix if the children really arrive as descriptors"
        );

        let reopened = Arc::new(rewrite_and_republish(&dataset, vec![reordered]).await);
        let expected_a = (0..rows)
            .map(|row| named_payload("a", row))
            .collect::<Vec<_>>();
        let expected_b = (0..rows)
            .map(|row| named_payload("b", row))
            .collect::<Vec<_>>();
        assert_eq!(
            blob_payloads(&reopened, "info.a", rows).await,
            expected_a,
            "child 'a' read back another child's payload"
        );
        assert_eq!(blob_payloads(&reopened, "info.b", rows).await, expected_b);
    }

    /// A projection of only one blob child of a struct: the remaining child keeps
    /// its own payload, and nothing is resolved against the dropped child's id.
    #[tokio::test]
    async fn a_partial_struct_projection_of_a_blob_child_is_handled_explicitly() {
        let dir = TempStrDir::default();
        let rows = 3;
        let dataset = struct_of_two_blobs_dataset(&dir, rows).await;

        let batches = scan_with_row_addresses(&dataset).await;
        let projected = with_struct_children_kept(&batches[0], "info", &["b"]);
        let plan =
            BlobRewritePlan::try_new(dataset.clone(), projected.schema().as_ref(), false).unwrap();
        let transformed = plan.transform(projected).await.unwrap();
        assert_eq!(transformed.schema(), *plan.output_schema());
        assert_eq!(
            BlobV2Layout::classify(&nested_blob_fields(
                plan.output_schema().as_ref(),
                "info",
                &["b"]
            )),
            Some(BlobV2Layout::Logical),
            "the projected child must still be rewritten"
        );

        let payloads = nested_blob_payloads(&transformed, "info", "b");
        let expected = (0..rows)
            .map(|row| named_payload("b", row))
            .collect::<Vec<_>>();
        assert_eq!(payloads, expected);
    }

    /// The inline payload bytes of a logical blob struct nested under a struct column.
    fn nested_blob_payloads(batch: &RecordBatch, column: &str, child: &str) -> Vec<Vec<u8>> {
        let outer = batch.column_by_name(column).unwrap().as_struct();
        let blob = outer.column_by_name(child).unwrap().as_struct();
        let data = blob.column(0).as_binary::<i64>();
        (0..data.len())
            .map(|row| data.value(row).to_vec())
            .collect()
    }

    /// Projecting away the blob child of a struct leaves a column with nothing to
    /// rewrite, which must pass through unchanged rather than be re-bound.
    #[tokio::test]
    async fn a_partial_struct_projection_of_a_non_blob_sibling_passes_through() {
        let dir = TempStrDir::default();
        let rows = 3;
        let children: Fields = vec![
            blob_field("a", true),
            Field::new("n", DataType::Int32, true),
        ]
        .into();
        let column: ArrayRef = Arc::new(
            StructArray::try_new(
                children.clone(),
                vec![
                    named_blob_values("a", rows),
                    Arc::new(Int32Array::from_iter_values(0..rows as i32)),
                ],
                None,
            )
            .unwrap(),
        );
        let dataset = nested_dataset(
            &dir,
            Field::new("info", DataType::Struct(children), true),
            column,
        )
        .await;

        let batches = scan_with_row_addresses(&dataset).await;
        let projected = with_struct_children_kept(&batches[0], "info", &["n"]);
        let plan =
            BlobRewritePlan::try_new(dataset.clone(), projected.schema().as_ref(), true).unwrap();
        let transformed = plan.transform(projected.clone()).await.unwrap();
        assert_eq!(transformed, projected);
    }

    /// A struct child the dataset does not have carries no field id, so it is
    /// refused instead of resolved against a sibling's id.
    #[tokio::test]
    async fn a_struct_child_unknown_to_the_dataset_is_refused() {
        let dir = TempStrDir::default();
        let rows = 3;
        let dataset = struct_of_two_blobs_dataset(&dir, rows).await;

        let batches = scan_with_row_addresses(&dataset).await;
        // A copy of child 'b' under a name the dataset does not have: the shape
        // is rewritable, only the binding is missing.
        let outer = batches[0].column_by_name("info").unwrap().as_struct();
        let (b_idx, b_field) = outer.fields().find("b").unwrap();
        let extended = with_extra_struct_child(
            &batches[0],
            "info",
            b_field
                .as_ref()
                .clone()
                .with_name("extra")
                .with_nullable(true),
            outer.column(b_idx).clone(),
        );
        let error = BlobRewritePlan::try_new(dataset.clone(), extended.schema().as_ref(), false)
            .unwrap_err();
        assert!(
            matches!(error, Error::InvalidInput { .. }),
            "unexpected error variant: {error:?}"
        );
        assert!(error.to_string().contains("info.extra"), "{error}");
    }

    /// The full struct in the dataset's own child order is the shape every
    /// in-crate caller builds, and it keeps each child's payload.
    #[tokio::test]
    async fn a_full_dataset_ordered_struct_keeps_every_child_payload() {
        let dir = TempStrDir::default();
        let rows = 3;
        let dataset = struct_of_two_blobs_dataset(&dir, rows).await;

        let batches = scan_with_row_addresses(&dataset).await;
        assert_eq!(
            batches[0]
                .column_by_name("info")
                .unwrap()
                .as_struct()
                .column_names(),
            vec!["a", "b"],
            "the scan must hand back the dataset's own child order"
        );

        let reopened = Arc::new(rewrite_and_republish(&dataset, batches).await);
        let expected_a = (0..rows)
            .map(|row| named_payload("a", row))
            .collect::<Vec<_>>();
        let expected_b = (0..rows)
            .map(|row| named_payload("b", row))
            .collect::<Vec<_>>();
        assert_eq!(blob_payloads(&reopened, "info.a", rows).await, expected_a);
        assert_eq!(blob_payloads(&reopened, "info.b", rows).await, expected_b);
    }
}
