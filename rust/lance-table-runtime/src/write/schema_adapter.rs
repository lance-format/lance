// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, RecordBatchIterator, RecordBatchReader};
use arrow_schema::{
    DataType, Field as ArrowField, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef,
};
use lance_arrow::json::{arrow_json_to_lance_json, convert_json_columns, has_arrow_json_fields};
use lance_core::Result;

/// Returns the physical field when it contains view arrays, including nested ones.
fn physical_field(field: &ArrowField) -> Option<ArrowField> {
    let data_type = match field.data_type() {
        DataType::Utf8View => DataType::Utf8,
        DataType::BinaryView => DataType::Binary,
        DataType::Struct(fields) => {
            let mut physical_fields = None;
            for (index, child) in fields.iter().enumerate() {
                if let Some(physical) = physical_field(child) {
                    physical_fields.get_or_insert_with(|| fields.to_vec())[index] =
                        Arc::new(physical);
                }
            }
            DataType::Struct(physical_fields?.into())
        }
        DataType::List(child) => DataType::List(Arc::new(physical_field(child)?)),
        DataType::LargeList(child) => DataType::LargeList(Arc::new(physical_field(child)?)),
        DataType::FixedSizeList(child, size) => {
            DataType::FixedSizeList(Arc::new(physical_field(child)?), *size)
        }
        DataType::Map(entries, sorted) => {
            DataType::Map(Arc::new(physical_field(entries)?), *sorted)
        }
        _ => return None,
    };
    Some(field.clone().with_data_type(data_type))
}

/// Cast view arrays to offset arrays, preserving nested fields and extension metadata.
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
/// stores: Arrow JSON text ↔ Lance JSONB, and view arrays → offset
/// arrays.
///
/// Table writes convert once in the data file writer. The physical methods
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
            let field = if has_arrow_json_fields(field) {
                Arc::new(arrow_json_to_lance_json(field))
            } else {
                Arc::clone(field)
            };
            new_fields.push(physical_field(&field).map(Arc::new).unwrap_or(field));
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Array, StringArray, StringViewArray, StructArray};
    use lance_arrow::ARROW_EXT_NAME_KEY;
    use lance_arrow::json::ARROW_JSON_EXT_NAME;

    #[test]
    fn nested_views_and_json_reader_schema_matches_batches() {
        let fields = vec![
            Arc::new(ArrowField::new("name", DataType::Utf8View, true)),
            Arc::new(
                ArrowField::new("document", DataType::Utf8, true).with_metadata(
                    [(
                        ARROW_EXT_NAME_KEY.to_string(),
                        ARROW_JSON_EXT_NAME.to_string(),
                    )]
                    .into(),
                ),
            ),
        ];
        let values = StructArray::new(
            fields.into(),
            vec![
                Arc::new(StringViewArray::from(vec![
                    Some("a sufficiently long view"),
                    None,
                ])),
                Arc::new(StringArray::from(vec![Some(r#"{"a":1}"#), None])),
            ],
            None,
        );
        let field = ArrowField::new("data", values.data_type().clone(), true)
            .with_metadata([("custom".into(), "preserved".into())].into());
        let schema = Arc::new(ArrowSchema::new(vec![field]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(values)]).unwrap();
        let adapter = SchemaAdapter::new(schema.clone());
        assert!(adapter.requires_physical_conversion());
        let mut reader = adapter.to_physical_reader(Box::new(RecordBatchIterator::new(
            [Ok(batch.clone())],
            schema,
        )));
        let converted = reader.next().unwrap().unwrap();
        assert_eq!(converted.schema(), reader.schema());
        assert_eq!(converted, adapter.to_physical_batch(batch).unwrap());
        let DataType::Struct(fields) = converted.schema().field(0).data_type().clone() else {
            panic!("expected a struct");
        };
        assert_eq!(fields[0].data_type(), &DataType::Utf8);
        assert_eq!(fields[1].data_type(), &DataType::LargeBinary);
        assert_eq!(
            converted.schema().field(0).metadata()["custom"],
            "preserved"
        );
        assert!(!SchemaAdapter::new(converted.schema()).requires_physical_conversion());
    }
}
