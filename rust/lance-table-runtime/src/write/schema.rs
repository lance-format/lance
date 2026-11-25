// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;

use arrow_schema::DataType;
use lance_arrow::{
    ARROW_EXT_NAME_KEY, BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY,
    BLOB_INLINE_SIZE_THRESHOLD_META_KEY, BLOB_META_KEY, BLOB_PACK_FILE_SIZE_THRESHOLD_META_KEY,
    BLOB_V2_EXT_NAME,
};
use lance_core::{
    Error, Result,
    datatypes::{
        Field, LogicalType, NullabilityComparison, OnMissing, OnTypeMismatch, Schema,
        SchemaCompareOptions,
    },
};
use lance_file::version::ConcreteFileVersion;
use lance_table::transaction::{Operation, resolve_arrow_field_ids};

use super::file::writes_blob_v2;
use super::{WriteMode, WriteParams};
use crate::Table;
use crate::blob::thresholds::{
    blob_dedicated_threshold_from_metadata, blob_inline_threshold_from_metadata,
    blob_pack_file_threshold_from_metadata,
};

fn validate_blob_threshold_metadata_for_append(
    input_schema: &Schema,
    dataset_schema: &Schema,
) -> Result<()> {
    for input_field in &input_schema.fields {
        let Some(dataset_field) = dataset_schema.field(&input_field.name) else {
            continue;
        };
        validate_blob_threshold_metadata_for_field_recursive(input_field, dataset_field)?;
    }

    Ok(())
}

fn validate_blob_threshold_metadata_for_field_recursive(
    input_field: &lance_core::datatypes::Field,
    dataset_field: &lance_core::datatypes::Field,
) -> Result<()> {
    let input_is_blob_v2 = input_field
        .metadata
        .get(ARROW_EXT_NAME_KEY)
        .is_some_and(|extension_name| extension_name == BLOB_V2_EXT_NAME);
    let dataset_is_blob_v2 = dataset_field
        .metadata
        .get(ARROW_EXT_NAME_KEY)
        .is_some_and(|extension_name| extension_name == BLOB_V2_EXT_NAME);
    if input_is_blob_v2 || dataset_is_blob_v2 {
        for (key, read_threshold) in [
            (
                BLOB_INLINE_SIZE_THRESHOLD_META_KEY,
                blob_inline_threshold_from_metadata
                    as fn(&HashMap<String, String>, &str) -> Result<usize>,
            ),
            (
                BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY,
                blob_dedicated_threshold_from_metadata,
            ),
            (
                BLOB_PACK_FILE_SIZE_THRESHOLD_META_KEY,
                blob_pack_file_threshold_from_metadata,
            ),
        ] {
            if !input_field.metadata.contains_key(key) {
                continue;
            }
            let input_value = read_threshold(&input_field.metadata, &input_field.name)?;
            let dataset_value = read_threshold(&dataset_field.metadata, &dataset_field.name)?;
            if input_value != dataset_value {
                return Err(Error::invalid_input(format!(
                    "Cannot append data with blob threshold metadata {key}={input_value} for \
                     field '{}'; the dataset schema has effective value {dataset_value}. Blob \
                     thresholds for existing columns are stored in the dataset schema.",
                    input_field.name,
                )));
            }
        }
    }

    for input_child in &input_field.children {
        let Some(dataset_child) = dataset_field.child(&input_child.name) else {
            continue;
        };
        validate_blob_threshold_metadata_for_field_recursive(input_child, dataset_child)?;
    }

    Ok(())
}

pub fn promote_legacy_blob_schema(schema: &Schema) -> Result<Schema> {
    let mut schema = schema.clone();
    for field in &mut schema.fields {
        field.promote_blob_v2()?;
    }
    schema.try_set_field_id(schema.max_field_id())?;
    Ok(schema)
}

/// Return promoted copies of the input and target schemas for compatibility
/// checks, reconciling equivalent Blob list tags for Blob v2 writes.
pub fn blob_write_comparison_schemas(
    input: &Schema,
    existing_schema: &Schema,
    write_version: ConcreteFileVersion,
) -> Result<(Schema, Schema)> {
    fn has_blob_v2_item(field: &Field) -> bool {
        matches!(
            field.children.as_slice(),
            [item] if item.logical_type.is_struct() && item.is_blob_v2()
        )
    }

    fn normalize_list_types(input: &mut Field, target: &mut Field) -> Result<()> {
        let same_list_kind = (input.logical_type.is_list() && target.logical_type.is_list())
            || (input.logical_type.is_large_list() && target.logical_type.is_large_list());
        // Blob promotion changes the item type, while the parent list can
        // retain its original tag. Reconcile tags only when both items are
        // Blob v2 after promotion and the two lists have the same offset width.
        if input.logical_type != target.logical_type
            && same_list_kind
            && has_blob_v2_item(input)
            && has_blob_v2_item(target)
        {
            let logical_type = LogicalType::try_from(&target.data_type())?;
            input.logical_type = logical_type.clone();
            target.logical_type = logical_type;
        }
        for input_child in &mut input.children {
            if let Some(target_child) = target.child_mut(&input_child.name) {
                normalize_list_types(input_child, target_child)?;
            }
        }
        Ok(())
    }

    let mut input = promote_legacy_blob_schema(input)?;
    let mut target = promote_legacy_blob_schema(existing_schema)?;
    if writes_blob_v2(write_version) {
        for input_field in &mut input.fields {
            if let Some(target_field) = target
                .fields
                .iter_mut()
                .find(|f| f.name == input_field.name)
            {
                normalize_list_types(input_field, target_field)?;
            }
        }
    }
    Ok((input, target))
}

use crate::commit::assign_new_table_field_ids;

pub(super) fn prepare_write_schema(
    dataset: Option<&Table>,
    normalized_converted_schema: Schema,
    params: &WriteParams,
    version: ConcreteFileVersion,
    inherited_field_ids: bool,
) -> Result<Schema> {
    let schema = if dataset.is_none() {
        let mut schema = normalized_converted_schema;
        assign_new_table_field_ids(&mut schema, inherited_field_ids)?;
        schema
    } else if let Some(dataset) = dataset
        && matches!(params.mode, WriteMode::Append | WriteMode::Create)
    {
        let mut schema_compare_options = schema_compare_options(version);
        schema_compare_options.compare_nullability = NullabilityComparison::Ignore;
        schema_compare_options.allow_missing_if_nullable = true;
        schema_compare_options.ignore_field_order = true;
        validate_blob_threshold_metadata_for_append(
            &normalized_converted_schema,
            dataset.schema(),
        )?;
        if normalized_converted_schema
            .check_compatible(dataset.schema(), &schema_compare_options)
            .is_ok()
        {
            return dataset.schema().project_by_schema(
                &normalized_converted_schema,
                OnMissing::Error,
                OnTypeMismatch::Error,
            );
        }
        let (comparison_schema, dataset_schema) =
            blob_write_comparison_schemas(&normalized_converted_schema, dataset.schema(), version)?;
        comparison_schema.check_compatible(&dataset_schema, &schema_compare_options)?;
        let mut projected = dataset_schema.project_by_schema(
            &comparison_schema,
            OnMissing::Error,
            OnTypeMismatch::Error,
        )?;
        // Inputs for 2.2+ were already promoted before schema preparation.
        // Remaining byte inputs retain the legacy physical layout and the table's field IDs.
        fn restore_legacy_blob_inputs(
            field: &mut lance_core::datatypes::Field,
            input: &lance_core::datatypes::Field,
        ) {
            if input.is_blob() && !input.is_blob_v2() {
                field.logical_type = input.logical_type.clone();
                field.children = input.children.clone();
                field.metadata = input.metadata.clone();
                field.encoding = input.encoding.clone();
            } else if !field.is_blob() {
                for child in &mut field.children {
                    if let Some(input) =
                        input.children.iter().find(|input| input.name == child.name)
                    {
                        restore_legacy_blob_inputs(child, input);
                    }
                }
            }
        }
        for field in &mut projected.fields {
            if let Some(input) = normalized_converted_schema
                .fields
                .iter()
                .find(|input| input.name == field.name)
            {
                restore_legacy_blob_inputs(field, input);
            }
        }
        projected
    } else if let Some(dataset) = dataset
        && matches!(params.mode, WriteMode::Overwrite)
        && dataset.manifest.uses_non_reusable_field_ids()
    {
        // Uncommitted fragment APIs return files without the schema used to
        // write them, so their mappings must already use commit-time IDs.
        // Resolve positional Arrow IDs before the schema enters the writer.
        let mut operation = Operation::Overwrite {
            fragments: Vec::new(),
            schema: normalized_converted_schema,
            config_upsert_values: None,
            initial_bases: None,
        };
        resolve_arrow_field_ids(Some(&dataset.manifest), &mut operation)?;
        match operation {
            Operation::Overwrite { schema, .. } => schema,
            _ => {
                return Err(Error::internal(
                    "Non-reusable field-ID canonicalization changed an Overwrite operation",
                ));
            }
        }
    } else {
        normalized_converted_schema
    };
    Ok(schema)
}

pub(super) fn validate_legacy_blob_write_schema(
    schema: &Schema,
    version_debug: &str,
) -> Result<()> {
    if schema.fields_pre_order().any(|field| field.is_blob_v2()) {
        return Err(Error::invalid_input(format!(
            "Blob v2 requires file version >= 2.2 (got {version_debug})"
        )));
    }
    Ok(())
}

pub(super) fn validate_blob_v2_write_schema(schema: &Schema) -> Result<()> {
    if let Some(blob_field_path) = legacy_blob_field_path(schema) {
        return Err(Error::invalid_input(format!(
            "Legacy blob columns (field metadata key {BLOB_META_KEY:?}) are not supported for file version >= 2.2. Found legacy blob field: {blob_field_path}. Use the blob v2 extension type (ARROW:extension:name = \"lance.blob.v2\") and the new blob APIs (e.g. lance::blob::blob_field / lance::blob::BlobArrayBuilder)."
        )));
    }
    Ok(())
}

fn legacy_blob_field_path(schema: &Schema) -> Option<String> {
    schema
        .fields_pre_order()
        .find(|field| field.metadata.contains_key(BLOB_META_KEY))
        .map(|field| {
            schema
                .field_path(field.id)
                .unwrap_or_else(|_| field.name.clone())
        })
}

pub fn schema_compare_options(version: ConcreteFileVersion) -> SchemaCompareOptions {
    match version {
        ConcreteFileVersion::V1 => SchemaCompareOptions {
            compare_dictionary: true,
            ..Default::default()
        },
        ConcreteFileVersion::V2_0
        | ConcreteFileVersion::V2_1
        | ConcreteFileVersion::V2_2
        | ConcreteFileVersion::V2_3 => SchemaCompareOptions::default(),
    }
}

/// Take the hidden row lineage columns out of a write schema.
///
/// A hidden column is a top-level field that already carries the reserved id
/// of its name, as the writers that spill lineage into a data file assign it.
/// A user column that only shares the name has a non-negative id and stays a
/// user column, to be checked against the dataset schema like any other.
pub(super) fn split_row_lineage_fields(schema: Schema) -> Result<(Schema, Vec<Field>)> {
    let (lineage, user): (Vec<_>, Vec<_>) = schema
        .fields
        .into_iter()
        .partition(|field| lance_core::row_lineage_field_id(&field.name) == Some(field.id));
    // The readers decode these columns as non-nullable `UInt64`, whatever the
    // written type, so any other type would only fail once read back.
    if let Some(field) = lineage
        .iter()
        .find(|field| field.nullable || field.data_type() != DataType::UInt64)
    {
        return Err(Error::internal(format!(
            "hidden row lineage column {} (field id {}) must be a non-nullable UInt64, got \
             {} (nullable: {})",
            field.name,
            field.id,
            field.data_type(),
            field.nullable
        )));
    }
    Ok((
        Schema {
            fields: user,
            metadata: schema.metadata,
        },
        lineage,
    ))
}

/// Zero row limits cannot be honored: a zero `max_rows_per_file` panics when
/// a V2 stream is split into files, and on V1 a zero `max_rows_per_group`
/// chunks the input into no batches, so nothing is written.
pub fn validate_row_limits(version: ConcreteFileVersion, params: &WriteParams) -> Result<()> {
    if params.max_rows_per_file == 0 {
        return Err(Error::invalid_input(
            "max_rows_per_file must be greater than zero",
        ));
    }
    if params.max_rows_per_file > u32::MAX as usize {
        return Err(Error::invalid_input(format!(
            "max_rows_per_file cannot exceed the fragment row capacity of {} rows, got {}",
            u32::MAX,
            params.max_rows_per_file
        )));
    }
    if matches!(version, ConcreteFileVersion::V1) && params.max_rows_per_group == 0 {
        return Err(Error::invalid_input(
            "max_rows_per_group must be greater than zero",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::Field as ArrowField;
    use lance_core::{ROW_CREATED_AT_VERSION, ROW_ID, ROW_ID_FIELD_ID};

    fn lance_field(name: &str, data_type: DataType, nullable: bool, id: i32) -> Field {
        let mut field = Field::try_from(&ArrowField::new(name, data_type, nullable)).unwrap();
        field.id = id;
        field
    }
    use rstest::rstest;

    #[rstest]
    #[case::uint64(DataType::UInt64, false, true)]
    #[case::not_uint64(DataType::Int64, false, false)]
    #[case::nullable(DataType::UInt64, true, false)]
    fn split_row_lineage_fields_keys_on_reserved_ids(
        #[case] data_type: DataType,
        #[case] nullable: bool,
        #[case] valid: bool,
    ) {
        let key = lance_field("i", DataType::Int32, false, 0);
        // A user column that only shares a lineage column's name keeps its own
        // id, so it stays with the fields checked against the dataset schema.
        let named_like_lineage = lance_field(ROW_CREATED_AT_VERSION, DataType::UInt64, true, 1);
        let hidden = lance_field(ROW_ID, data_type, nullable, ROW_ID_FIELD_ID);
        let schema = Schema {
            fields: vec![key.clone(), named_like_lineage.clone(), hidden.clone()],
            metadata: HashMap::new(),
        };

        let result = split_row_lineage_fields(schema);
        if !valid {
            let error = result.unwrap_err();
            assert!(matches!(error, Error::Internal { .. }), "{error}");
            assert!(error.to_string().contains("non-nullable UInt64"), "{error}");
            return;
        }
        let (user, lineage) = result.unwrap();
        assert_eq!(user.fields, vec![key, named_like_lineage]);
        // Split out with its reserved id intact.
        assert_eq!(lineage, vec![hidden]);
    }

    #[test]
    fn rejects_file_limit_above_fragment_row_capacity() {
        let params = WriteParams {
            max_rows_per_file: u32::MAX as usize + 1,
            ..Default::default()
        };

        let error = validate_row_limits(ConcreteFileVersion::V2_0, &params).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error
                .to_string()
                .contains("cannot exceed the fragment row capacity"),
            "{error}"
        );
    }
}
