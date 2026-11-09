// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Rewrite Blob v2 batches into the logical writer representation.
//!
//! The transformation is shared by scans, updates, merge-insert, managed
//! clones, and compaction. Nested structs and lists retain their own nulls and
//! offsets while Blob v2 leaves are converted from stored descriptors to the
//! payload representation accepted by writers.

use std::sync::Arc;

use arrow::datatypes::{UInt8Type, UInt32Type, UInt64Type};
use arrow_array::builder::{LargeBinaryBuilder, PrimitiveBuilder, StringBuilder};
use arrow_array::cast::AsArray;
use arrow_array::{
    Array, ArrayRef, GenericListArray, OffsetSizeTrait, RecordBatch, RecordBatchOptions,
    StructArray, UInt32Array,
};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_schema::{
    DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema, SchemaRef,
};
use futures::FutureExt;
use futures::future::BoxFuture;
use lance_arrow::{list::ListArrayExt, r#struct::StructArrayExt};
use lance_core::datatypes::{
    BLOB_V2_LOGICAL_FIELDS, BLOB_V2_LOGICAL_TYPE, BlobKind, BlobV2Layout, Field as LanceField,
};
use lance_core::{Error, Result};

use super::{collect_blob_v2_descriptor_files, preserve_managed_descriptors};
use crate::Dataset;

/// Classification for one blob v2 row during rewrite.
///
/// - `Null`: NULL row.
/// - `External`: External blob referenced by URI.
/// - `DataBlob`: Inline/Packed/Dedicated blob stored in Lance files.
enum RowClass {
    Null,
    External,
    DataBlob,
}

/// Column views for the 5 fields in a blob v2 descriptor struct.
struct BlobV2Descriptor<'a> {
    kind_col: &'a arrow::array::UInt8Array,
    position_col: &'a arrow::array::UInt64Array,
    size_col: &'a arrow::array::UInt64Array,
    blob_uri_col: &'a arrow::array::StringArray,
    blob_id_col: &'a arrow::array::UInt32Array,
}

impl<'a> BlobV2Descriptor<'a> {
    /// Extract the 5 descriptor arrays from a blob v2 descriptor struct array.
    fn try_from_struct(struct_arr: &'a StructArray, column_name: &str) -> Result<Self> {
        if BlobV2Layout::classify(struct_arr.fields()) != Some(BlobV2Layout::Descriptor) {
            let actual = BlobV2Layout::classify(struct_arr.fields())
                .map(|layout| layout.to_string())
                .unwrap_or_else(|| format!("unrecognized ({:?})", struct_arr.fields()));
            return Err(Error::invalid_input(format!(
                "Blob v2 column '{column_name}' has {actual} layout; expected descriptor layout before conversion to logical"
            )));
        }
        let kind_col = struct_arr
            .column_by_name("kind")
            .ok_or_else(|| {
                Error::internal(format!(
                    "Blob v2 descriptor for column '{}' missing `kind` field",
                    column_name
                ))
            })?
            .as_primitive::<UInt8Type>();
        let position_col = struct_arr
            .column_by_name("position")
            .ok_or_else(|| {
                Error::internal(format!(
                    "Blob v2 descriptor for column '{}' missing `position` field",
                    column_name
                ))
            })?
            .as_primitive::<UInt64Type>();
        let size_col = struct_arr
            .column_by_name("size")
            .ok_or_else(|| {
                Error::internal(format!(
                    "Blob v2 descriptor for column '{}' missing `size` field",
                    column_name
                ))
            })?
            .as_primitive::<UInt64Type>();
        let blob_uri_col = struct_arr
            .column_by_name("blob_uri")
            .ok_or_else(|| {
                Error::internal(format!(
                    "Blob v2 descriptor for column '{}' missing `blob_uri` field",
                    column_name
                ))
            })?
            .as_string::<i32>();
        let blob_id_col = struct_arr
            .column_by_name("blob_id")
            .ok_or_else(|| {
                Error::internal(format!(
                    "Blob v2 descriptor for column '{}' missing `blob_id` field",
                    column_name
                ))
            })?
            .as_primitive::<UInt32Type>();
        Ok(Self {
            kind_col,
            position_col,
            size_col,
            blob_uri_col,
            blob_id_col,
        })
    }
}

/// Result of row classification for a blob v2 rewrite.
struct RowClassification {
    row_classes: Vec<RowClass>,
    data_blob_indices: Vec<usize>,
}

/// Classify each row of a blob v2 column as Null, External, or DataBlob.
fn classify_rows(
    struct_arr: &StructArray,
    descriptor: &BlobV2Descriptor<'_>,
    column_name: &str,
) -> Result<RowClassification> {
    let num_rows = struct_arr.len();
    let mut row_classes = Vec::with_capacity(num_rows);
    let mut data_blob_indices = Vec::with_capacity(num_rows);

    for i in 0..num_rows {
        if struct_arr.is_null(i) || descriptor.kind_col.is_null(i) {
            row_classes.push(RowClass::Null);
        } else {
            let kind = BlobKind::try_from(descriptor.kind_col.value(i)).map_err(|e| {
                Error::internal(format!(
                    "Blob v2 column '{}' has invalid kind at row {}: {e}",
                    column_name, i
                ))
            })?;
            if kind == BlobKind::External {
                row_classes.push(RowClass::External);
            } else {
                row_classes.push(RowClass::DataBlob);
                data_blob_indices.push(i);
            }
        }
    }

    Ok(RowClassification {
        row_classes,
        data_blob_indices,
    })
}

/// The URI of the object at `relative_path` under the base at `base_uri`, each
/// path segment escaped so characters such as `#` and `?` stay in the name.
fn external_blob_uri(base_uri: &str, relative_path: &str) -> Result<String> {
    let mut url = lance_io::object_store::uri_to_url(base_uri)?;
    url.path_segments_mut()
        .map_err(|_| {
            Error::invalid_input(format!(
                "External blob base '{base_uri}' cannot hold an object path"
            ))
        })?
        .pop_if_empty()
        .extend(relative_path.split('/'));
    Ok(url.to_string())
}

/// Convert a blob v2 descriptor into the logical writer representation.
///
/// Reads blob data lazily using row addresses to avoid materializing all blob
/// payloads in memory at once.
async fn descriptor_to_logical_blob_array(
    dataset: &Arc<Dataset>,
    blob_field_id: u32,
    struct_arr: &StructArray,
    descriptor: &BlobV2Descriptor<'_>,
    classification: &RowClassification,
    row_addrs: &[u64],
    field_name: &str,
) -> Result<StructArray> {
    if struct_arr.len() != row_addrs.len() {
        return Err(Error::internal(format!(
            "Blob v2 field '{}' row count {} did not match row address count {}",
            field_name,
            struct_arr.len(),
            row_addrs.len()
        )));
    }
    let blob_files = if classification.data_blob_indices.is_empty() {
        Vec::new()
    } else {
        let indices = classification
            .data_blob_indices
            .iter()
            .map(|index| {
                u32::try_from(*index).map_err(|_| {
                    Error::internal(format!(
                        "Blob v2 row index {} in field '{}' does not fit in u32",
                        index, field_name
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let indices = UInt32Array::from(indices);
        let descriptions = arrow_select::take::take(struct_arr, &indices, None)?;
        let descriptions = descriptions.as_struct();
        let data_row_addrs = classification
            .data_blob_indices
            .iter()
            .map(|index| row_addrs[*index])
            .collect::<Vec<_>>();
        collect_blob_v2_descriptor_files(dataset, blob_field_id, descriptions, &data_row_addrs)
            .await?
    };

    let data_capacity =
        classification
            .data_blob_indices
            .iter()
            .try_fold(0usize, |data_capacity, row_idx| {
                if descriptor.size_col.is_null(*row_idx) {
                    return Err(Error::internal(format!(
                        "Non-null blob row {} in field '{}' is missing its size",
                        row_idx, field_name
                    )));
                }
                let size = usize::try_from(descriptor.size_col.value(*row_idx)).map_err(|_| {
                    Error::internal(format!(
                        "Blob size {} at row {} in field '{}' does not fit in usize",
                        descriptor.size_col.value(*row_idx),
                        row_idx,
                        field_name
                    ))
                })?;
                data_capacity.checked_add(size).ok_or_else(|| {
                    Error::internal(format!(
                        "Total blob size in field '{}' exceeds usize",
                        field_name
                    ))
                })
            })?;

    let num_rows = struct_arr.len();
    let mut data_builder = LargeBinaryBuilder::with_capacity(num_rows, data_capacity);
    let mut uri_builder = StringBuilder::with_capacity(num_rows, 0);
    let mut out_position_builder = PrimitiveBuilder::<UInt64Type>::with_capacity(num_rows);
    let mut out_size_builder = PrimitiveBuilder::<UInt64Type>::with_capacity(num_rows);

    let mut blob_file_idx = 0;
    #[allow(clippy::needless_range_loop)]
    for i in 0..num_rows {
        match classification.row_classes[i] {
            RowClass::Null => {
                data_builder.append_null();
                uri_builder.append_null();
                out_position_builder.append_null();
                out_size_builder.append_null();
            }
            RowClass::External => {
                data_builder.append_null();
                let base_id = descriptor.blob_id_col.value(i);
                let uri_val = descriptor.blob_uri_col.value(i);
                if base_id == 0 {
                    uri_builder.append_value(uri_val);
                } else {
                    let base = dataset.manifest().base_paths.get(&base_id).ok_or_else(|| {
                        Error::internal(format!(
                            "External blob in column '{}' references unknown base_id {}",
                            field_name, base_id
                        ))
                    })?;
                    uri_builder.append_value(external_blob_uri(&base.path, uri_val)?);
                }
                let position =
                    (!descriptor.position_col.is_null(i)).then(|| descriptor.position_col.value(i));
                let size = (!descriptor.size_col.is_null(i)).then(|| descriptor.size_col.value(i));
                if position == Some(0) && size == Some(0) {
                    // Stable descriptors use (0, 0) for the complete external object.
                    // Logical input represents the same value by omitting the range.
                    out_position_builder.append_null();
                    out_size_builder.append_null();
                } else {
                    out_position_builder.append_option(position);
                    out_size_builder.append_option(size);
                }
            }
            RowClass::DataBlob => {
                let blob_file = blob_files
                    .get(blob_file_idx)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| {
                        Error::internal(format!(
                            "Non-null blob row {} in field '{}' resolved to null",
                            i, field_name
                        ))
                    })?;
                let data = blob_file.read().await?;
                blob_file_idx += 1;
                data_builder.append_value(data.as_ref());
                uri_builder.append_null();
                out_position_builder.append_null();
                out_size_builder.append_null();
            }
        }
    }

    Ok(StructArray::try_new(
        BLOB_V2_LOGICAL_FIELDS.clone(),
        vec![
            Arc::new(data_builder.finish()),
            Arc::new(uri_builder.finish()),
            Arc::new(out_position_builder.finish()),
            Arc::new(out_size_builder.finish()),
        ],
        struct_arr.nulls().cloned(),
    )?)
}

fn transformed_arrow_field(field: &LanceField, data_type: ArrowDataType) -> Arc<ArrowField> {
    let arrow_field = ArrowField::from(field);
    arrow_field_with_data_type(&arrow_field, data_type)
}

fn arrow_field_with_data_type(field: &ArrowField, data_type: ArrowDataType) -> Arc<ArrowField> {
    Arc::new(
        ArrowField::new(field.name(), data_type, field.is_nullable())
            .with_metadata(field.metadata().clone()),
    )
}

pub fn field_contains_blob_v2(field: &LanceField) -> bool {
    field.is_blob_v2() || field.children.iter().any(field_contains_blob_v2)
}

enum BlobV2FieldRewritePlan {
    Passthrough {
        output_field: Arc<ArrowField>,
    },
    Blob {
        sidecars: Option<Arc<super::SidecarRewrite>>,
        field_id: u32,
        field_name: String,
        output_field: Arc<ArrowField>,
    },
    Struct {
        field_name: String,
        output_field: Arc<ArrowField>,
        children: Vec<Self>,
    },
    List {
        field_name: String,
        output_field: Arc<ArrowField>,
        child: Box<Self>,
    },
    LargeList {
        field_name: String,
        output_field: Arc<ArrowField>,
        child: Box<Self>,
    },
}

impl BlobV2FieldRewritePlan {
    fn passthrough(field: &ArrowField) -> Self {
        Self::Passthrough {
            output_field: Arc::new(field.clone()),
        }
    }

    fn try_new(
        field: &LanceField,
        input_field: &ArrowField,
        preserve_managed: bool,
    ) -> Result<Self> {
        if !field_contains_blob_v2(field) {
            return Ok(Self::passthrough(input_field));
        }

        if field.is_blob_v2() {
            let field_id = u32::try_from(field.id).map_err(|_| {
                Error::internal(format!(
                    "Blob v2 field id {} for '{}' does not fit in u32",
                    field.id, field.name
                ))
            })?;
            let ArrowDataType::Struct(input_children) = input_field.data_type() else {
                // Legacy blob bytes: the writer converts them to Blob v2.
                if matches!(
                    input_field.data_type(),
                    ArrowDataType::Binary | ArrowDataType::LargeBinary
                ) {
                    return Ok(Self::passthrough(input_field));
                }
                return Err(Error::invalid_input(format!(
                    "Blob v2 field '{}' has non-struct input type {:?}",
                    field.name,
                    input_field.data_type()
                )));
            };
            let output_field = match BlobV2Layout::classify(input_children) {
                Some(BlobV2Layout::Logical) => Arc::new(input_field.clone()),
                Some(BlobV2Layout::Descriptor) => transformed_arrow_field(
                    field,
                    if preserve_managed {
                        lance_core::datatypes::BLOB_V2_PREPARED_TYPE.clone()
                    } else {
                        BLOB_V2_LOGICAL_TYPE.clone()
                    },
                ),
                Some(actual) => {
                    return Err(Error::invalid_input(format!(
                        "Blob v2 field '{}' has {actual} input layout; expected logical or descriptor layout during rewrite",
                        field.name
                    )));
                }
                None => {
                    return Err(Error::invalid_input(format!(
                        "Blob v2 field '{}' has unrecognized input layout {:?}; expected logical or descriptor layout during rewrite",
                        field.name, input_children
                    )));
                }
            };
            return Ok(Self::Blob {
                sidecars: None,
                field_id,
                field_name: field.name.clone(),
                output_field,
            });
        }

        match (field.data_type(), input_field.data_type()) {
            (ArrowDataType::Struct(_), ArrowDataType::Struct(input_children)) => {
                if field.children.len() != input_children.len() {
                    return Err(Error::internal(format!(
                        "Struct field '{}' expected {} children in blob rewrite plan, got {}",
                        field.name,
                        field.children.len(),
                        input_children.len()
                    )));
                }
                let children = field
                    .children
                    .iter()
                    .zip(input_children.iter())
                    .map(|(child, input_child)| Self::try_new(child, input_child, preserve_managed))
                    .collect::<Result<Vec<_>>>()?;
                let output_children = children
                    .iter()
                    .map(|child| child.output_field().clone())
                    .collect::<Vec<_>>();
                Ok(Self::Struct {
                    field_name: field.name.clone(),
                    output_field: arrow_field_with_data_type(
                        input_field,
                        ArrowDataType::Struct(output_children.into()),
                    ),
                    children,
                })
            }
            (ArrowDataType::List(_), ArrowDataType::List(input_child)) => {
                let child = field.children.first().ok_or_else(|| {
                    Error::internal(format!(
                        "List field '{}' is missing its child in blob rewrite plan",
                        field.name
                    ))
                })?;
                let child = Box::new(Self::try_new(child, input_child, preserve_managed)?);
                Ok(Self::List {
                    field_name: field.name.clone(),
                    output_field: arrow_field_with_data_type(
                        input_field,
                        ArrowDataType::List(child.output_field().clone()),
                    ),
                    child,
                })
            }
            (ArrowDataType::LargeList(_), ArrowDataType::LargeList(input_child)) => {
                let child = field.children.first().ok_or_else(|| {
                    Error::internal(format!(
                        "Large list field '{}' is missing its child in blob rewrite plan",
                        field.name
                    ))
                })?;
                let child = Box::new(Self::try_new(child, input_child, preserve_managed)?);
                Ok(Self::LargeList {
                    field_name: field.name.clone(),
                    output_field: arrow_field_with_data_type(
                        input_field,
                        ArrowDataType::LargeList(child.output_field().clone()),
                    ),
                    child,
                })
            }
            (logical_type, input_type) => Err(Error::invalid_input(format!(
                "Field '{}' contains blob v2 descendants but logical type {:?} and input type {:?} do not form a supported rewrite container",
                field.name, logical_type, input_type
            ))),
        }
    }

    fn output_field(&self) -> &Arc<ArrowField> {
        match self {
            Self::Passthrough { output_field }
            | Self::Blob { output_field, .. }
            | Self::Struct { output_field, .. }
            | Self::List { output_field, .. }
            | Self::LargeList { output_field, .. } => output_field,
        }
    }

    fn transform<'a>(
        &'a self,
        dataset: &'a Arc<Dataset>,
        array: ArrayRef,
        row_addrs: Arc<[u64]>,
    ) -> BoxFuture<'a, Result<ArrayRef>> {
        async move {
            match self {
                Self::Passthrough { .. } => Ok(array),
                Self::Blob {
                    sidecars,
                    field_id,
                    field_name,
                    output_field,
                } => {
                    let struct_arr = array.as_struct();
                    match BlobV2Layout::classify(struct_arr.fields()) {
                        Some(BlobV2Layout::Logical) => Ok(array),
                        Some(BlobV2Layout::Descriptor) => {
                            if matches!(output_field.data_type(), ArrowDataType::Struct(fields) if BlobV2Layout::classify(fields) == Some(BlobV2Layout::Prepared)) {
                                if let Some(sidecars) = sidecars {
                                    return sidecars.prepare(dataset, *field_id, struct_arr, &row_addrs, output_field).await;
                                }
                                return preserve_managed_descriptors(dataset, *field_id, struct_arr, &row_addrs, output_field, None).await;
                            }
                            let descriptor =
                                BlobV2Descriptor::try_from_struct(struct_arr, field_name)?;
                            let classification =
                                classify_rows(struct_arr, &descriptor, field_name)?;
                            let logical = descriptor_to_logical_blob_array(
                                dataset,
                                *field_id,
                                struct_arr,
                                &descriptor,
                                &classification,
                                row_addrs.as_ref(),
                                field_name,
                            )
                            .await?;
                            Ok(Arc::new(logical) as ArrayRef)
                        }
                        Some(actual) => Err(Error::invalid_input(format!(
                            "Blob v2 field '{}' has {actual} layout; expected logical or descriptor layout during rewrite",
                            field_name
                        ))),
                        None => Err(Error::invalid_input(format!(
                            "Blob v2 field '{}' has unrecognized layout {:?}; expected logical or descriptor layout during rewrite",
                            field_name,
                            struct_arr.fields()
                        ))),
                    }
                }
                Self::Struct {
                    field_name,
                    output_field,
                    children,
                } => {
                    let struct_arr = array.as_struct().normalize_slicing()?;
                    let parent_nulls = struct_arr.nulls().cloned();
                    let struct_arr = struct_arr.pushdown_nulls()?;
                    if children.len() != struct_arr.num_columns() {
                        return Err(Error::internal(format!(
                            "Struct field '{}' expected {} children during blob rewrite, got {}",
                            field_name,
                            children.len(),
                            struct_arr.num_columns()
                        )));
                    }
                    let mut child_arrays = Vec::with_capacity(children.len());
                    for (child, child_array) in children.iter().zip(struct_arr.columns()) {
                        child_arrays.push(
                            child
                                .transform(dataset, child_array.clone(), row_addrs.clone())
                                .await?,
                        );
                    }
                    let ArrowDataType::Struct(output_fields) = output_field.data_type() else {
                        return Err(Error::internal(format!(
                            "Struct field '{}' rewrite plan has non-struct output type {:?}",
                            field_name,
                            output_field.data_type()
                        )));
                    };
                    Ok(Arc::new(StructArray::try_new(
                        output_fields.clone(),
                        child_arrays,
                        parent_nulls,
                    )?) as ArrayRef)
                }
                Self::List {
                    field_name, child, ..
                } => {
                    transform_blob_v2_list_array::<i32>(
                        dataset,
                        field_name,
                        child,
                        array.as_list::<i32>(),
                        row_addrs,
                    )
                    .await
                }
                Self::LargeList {
                    field_name, child, ..
                } => {
                    transform_blob_v2_list_array::<i64>(
                        dataset,
                        field_name,
                        child,
                        array.as_list::<i64>(),
                        row_addrs,
                    )
                    .await
                }
            }
        }
        .boxed()
    }
}

async fn transform_blob_v2_list_array<O: OffsetSizeTrait>(
    dataset: &Arc<Dataset>,
    field_name: &str,
    child: &BlobV2FieldRewritePlan,
    list_array: &GenericListArray<O>,
    row_addrs: Arc<[u64]>,
) -> Result<ArrayRef> {
    let list_array = if list_array.null_count() > 0 {
        list_array.filter_garbage_nulls()
    } else {
        list_array.clone()
    };
    let offsets = list_array.value_offsets();
    let values_start = offsets[0].as_usize();
    let values_end = offsets[list_array.len()].as_usize();
    if values_end < values_start {
        return Err(Error::internal(format!(
            "List field '{}' has invalid offsets during blob rewrite",
            field_name
        )));
    }

    let values_len = values_end - values_start;
    let mut normalized_offsets = Vec::with_capacity(list_array.len() + 1);
    normalized_offsets.push(O::usize_as(0));
    let mut child_row_addrs = Vec::with_capacity(values_len);
    for row_idx in 0..list_array.len() {
        let start = offsets[row_idx].as_usize();
        let end = offsets[row_idx + 1].as_usize();
        if end < start {
            return Err(Error::internal(format!(
                "List field '{}' has decreasing offsets during blob rewrite",
                field_name
            )));
        }
        let row_addr = row_addrs.get(row_idx).copied().ok_or_else(|| {
            Error::internal(format!(
                "List field '{}' row address count {} did not match row count {}",
                field_name,
                row_addrs.len(),
                list_array.len()
            ))
        })?;
        child_row_addrs.extend(std::iter::repeat_n(row_addr, end - start));
        normalized_offsets.push(O::usize_as(end - values_start));
    }

    let values = list_array.values().slice(values_start, values_len);
    let values = child
        .transform(dataset, values, Arc::<[u64]>::from(child_row_addrs))
        .await?;
    let list_array = GenericListArray::<O>::try_new(
        child.output_field().clone(),
        OffsetBuffer::new(ScalarBuffer::from(normalized_offsets)),
        values,
        list_array.nulls().cloned(),
    )?;
    Ok(Arc::new(list_array))
}

/// Reusable Blob v2 rewrite plan for new fragments of the same dataset.
///
/// Used by compaction to preserve payload references or materialize payload bytes.
pub struct BlobV2BatchRewritePlan {
    row_addr_idx: usize,
    columns: Vec<(usize, BlobV2FieldRewritePlan)>,
    output_schema: SchemaRef,
}

impl BlobV2BatchRewritePlan {
    /// Build a plan from the dataset schema and scanned batch schema.
    ///
    /// Input must include `_rowaddr`; Struct children must be complete and in
    /// dataset order. Scan legacy blobs as binary and Blob v2 as descriptors.
    /// `keep_row_addr` retains `_rowaddr` in the output.
    ///
    /// `preserve_managed` keeps payload references instead of materializing bytes.
    /// It requires Managed blobs and a V2_2+ writer. Write output to the same
    /// dataset's primary storage.
    pub fn try_new(
        schema: &lance_core::datatypes::Schema,
        input_schema: &ArrowSchema,
        keep_row_addr: bool,
        preserve_managed: bool,
    ) -> Result<Self> {
        let row_addr_idx = input_schema
            .column_with_name(lance_core::ROW_ADDR)
            .ok_or_else(|| {
                Error::internal(format!(
                    "_rowaddr column missing from batch for blob v2 rewrite, columns: {:?}",
                    input_schema
                        .fields()
                        .iter()
                        .map(|f| f.name())
                        .collect::<Vec<_>>()
                ))
            })?
            .0;
        let mut columns = Vec::with_capacity(input_schema.fields().len());
        let mut output_fields = Vec::with_capacity(input_schema.fields().len());
        for (column_idx, input_field) in input_schema.fields().iter().enumerate() {
            if input_field.name() == lance_core::ROW_ADDR && !keep_row_addr {
                continue;
            }
            let field_plan = if let Some(field) = schema.field(input_field.name()) {
                BlobV2FieldRewritePlan::try_new(field, input_field, preserve_managed)?
            } else {
                BlobV2FieldRewritePlan::passthrough(input_field)
            };
            output_fields.push(field_plan.output_field().as_ref().clone());
            columns.push((column_idx, field_plan));
        }

        Ok(Self {
            row_addr_idx,
            columns,
            output_schema: Arc::new(ArrowSchema::new_with_metadata(
                output_fields,
                input_schema.metadata().clone(),
            )),
        })
    }

    /// Build the prepared rewrite used by ordinary sidecar compaction.
    pub(in crate::dataset) fn with_sidecar_rewrite(
        mut self,
        sidecars: Arc<super::SidecarRewrite>,
    ) -> Self {
        fn configure(plan: &mut BlobV2FieldRewritePlan, sidecars: &Arc<super::SidecarRewrite>) {
            match plan {
                BlobV2FieldRewritePlan::Blob {
                    sidecars: context, ..
                } => *context = Some(sidecars.clone()),
                BlobV2FieldRewritePlan::Struct { children, .. } => {
                    for child in children {
                        configure(child, sidecars);
                    }
                }
                BlobV2FieldRewritePlan::List { child, .. }
                | BlobV2FieldRewritePlan::LargeList { child, .. } => configure(child, sidecars),
                BlobV2FieldRewritePlan::Passthrough { .. } => {}
            }
        }
        for (_, column) in &mut self.columns {
            configure(column, &sidecars);
        }
        self
    }

    /// Schema of rewritten batches, for opening the destination writer.
    pub fn output_schema(&self) -> &SchemaRef {
        &self.output_schema
    }

    /// Rewrite a batch using Blob payloads from `dataset`.
    ///
    /// `batch` must match the planned schema and contain valid, non-null `UInt64`
    /// row addresses from this dataset snapshot. These preconditions are not
    /// validated. Keep the snapshot alive until the output is committed.
    ///
    /// May read from storage. Batches can be transformed concurrently.
    pub async fn transform_batch(
        &self,
        dataset: &Arc<Dataset>,
        batch: RecordBatch,
    ) -> Result<RecordBatch> {
        let row_addrs: Arc<[u64]> = batch
            .column(self.row_addr_idx)
            .as_primitive::<UInt64Type>()
            .values()
            .iter()
            .copied()
            .collect::<Vec<_>>()
            .into();
        let mut output_columns = Vec::with_capacity(self.columns.len());
        for (column_idx, field_plan) in &self.columns {
            output_columns.push(
                field_plan
                    .transform(
                        dataset,
                        batch.column(*column_idx).clone(),
                        row_addrs.clone(),
                    )
                    .await?,
            );
        }
        // The row count keeps a batch with no output columns (an update that
        // assigns every column) the size of its input.
        Ok(RecordBatch::try_new_with_options(
            self.output_schema.clone(),
            output_columns,
            &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
        )?)
    }
}

pub async fn transform_blob_v2_batch(
    dataset: &Arc<Dataset>,
    schema: &lance_core::datatypes::Schema,
    batch: RecordBatch,
    keep_row_addr: bool,
) -> Result<RecordBatch> {
    let plan =
        BlobV2BatchRewritePlan::try_new(schema, batch.schema().as_ref(), keep_row_addr, false)?;
    plan.transform_batch(dataset, batch).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::{DataType, Field, Schema};

    #[test]
    fn test_blob_v2_rewrite_plan_skips_non_blob_list_subtree() {
        let items_field = Field::new(
            "items",
            DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
            true,
        );
        let info_field = Field::new(
            "info",
            DataType::Struct(
                vec![
                    Arc::new(items_field),
                    Arc::new(crate::blob_field("blob", true)),
                ]
                .into(),
            ),
            true,
        );
        let logical_arrow_schema = Schema::new(vec![info_field]);
        let logical_schema =
            lance_core::datatypes::Schema::try_from(&logical_arrow_schema).unwrap();
        let mut input_schema = logical_schema.clone();
        input_schema.fields[0].unload_blobs_recursive();
        let input_field = Field::from(&input_schema.fields[0]);

        let plan = BlobV2FieldRewritePlan::try_new(&logical_schema.fields[0], &input_field, false)
            .unwrap();
        let BlobV2FieldRewritePlan::Struct { children, .. } = &plan else {
            panic!("nested blob field should produce a struct rewrite plan");
        };
        assert!(matches!(
            children[0],
            BlobV2FieldRewritePlan::Passthrough { .. }
        ));
        assert!(matches!(children[1], BlobV2FieldRewritePlan::Blob { .. }));

        let DataType::Struct(output_children) = plan.output_field().data_type() else {
            panic!("nested blob rewrite plan should declare a struct output");
        };
        let DataType::Struct(input_children) = input_field.data_type() else {
            panic!("nested blob input should be a struct");
        };
        assert_eq!(output_children[0], input_children[0]);
        let DataType::Struct(blob_children) = output_children[1].data_type() else {
            panic!("blob rewrite plan should declare a struct blob output");
        };
        assert_eq!(
            BlobV2Layout::classify(blob_children),
            Some(BlobV2Layout::Logical)
        );
    }
}
