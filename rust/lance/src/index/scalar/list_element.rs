// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Binding a list-element index ([`lance_index::scalar::list_element`]) to the
//! current schema.
//!
//! The index's details name its key fields by Lance field ID, so that the index
//! follows a rename and is not fooled by another field taking a key's name. A
//! plugin only sees Arrow fields, which carry no IDs, so Lance records the IDs
//! when a segment is trained and, whenever it hands the details back, resolves
//! them to the names the fields have now.

use std::sync::Arc;

use lance_core::datatypes::Field;
use lance_index::pb::ListElementIndexDetails;
use lance_index::scalar::list_element::ListElementIndex;
use lance_index::scalar::{CreatedIndex, ScalarIndex};
use lance_table::format::IndexMetadata;
use prost::Message;

use crate::{Error, Result};

/// Whether `details` belong to a list-element index segment.
pub fn is_list_element(details: &prost_types::Any) -> bool {
    details.type_url.rsplit_once('/').is_some_and(|(_, name)| {
        name.eq_ignore_ascii_case("lance.index.pb.ListElementIndexDetails")
    })
}

fn decode(details: &prost_types::Any) -> Result<ListElementIndexDetails> {
    ListElementIndexDetails::decode(details.value.as_slice()).map_err(|err| {
        Error::corrupt_file_named(
            "ListElementIndexDetails",
            format!("cannot decode list-element index details: {err}"),
        )
    })
}

/// The direct children of the element struct of `column`, or `None` when it is
/// not a list (or large list) of structs.
fn element_fields(column: &Field) -> Option<&[Field]> {
    if !(column.logical_type.is_list() || column.logical_type.is_large_list()) {
        return None;
    }
    let item = column.children.first()?;
    item.logical_type
        .is_struct()
        .then_some(item.children.as_slice())
}

/// Record, in a segment the plugin has just trained on `column`, the field IDs
/// of the keys it names. Training validated the names against the same schema.
pub fn record_key_field_ids(column: &Field, created: &mut CreatedIndex) -> Result<()> {
    let mut details = decode(&created.index_details)?;
    let fields = element_fields(column).ok_or_else(|| {
        Error::invalid_input(format!(
            "a list-element index needs a list<struct> column, but '{}' is {}",
            column.name, column.logical_type
        ))
    })?;
    details.key_field_ids = details
        .keys
        .iter()
        .map(|key| {
            fields
                .iter()
                .find(|field| &field.name == key)
                .map(|field| field.id)
                .ok_or_else(|| {
                    Error::internal(format!(
                        "trained list-element key '{key}' is not a field of the elements of '{}'",
                        column.name
                    ))
                })
        })
        .collect::<Result<_>>()?;
    created.index_details = prost_types::Any::from_msg(&details)?;
    Ok(())
}

/// The details of a list-element segment on `column`, with each key field
/// resolved by ID among the direct children of the element struct and named as
/// it is now.
///
/// `None` when the segment no longer describes the column: a key field was
/// dropped or moved, its type changed, or its ID was never recorded. Queries
/// then scan the column instead of using the index.
pub fn bind_key_fields(
    column: &Field,
    details: &prost_types::Any,
) -> Result<Option<prost_types::Any>> {
    let mut decoded = decode(details)?;
    let Some(fields) = element_fields(column) else {
        return Ok(None);
    };
    if decoded.key_field_ids.is_empty() || decoded.key_field_ids.len() != decoded.key_types.len() {
        return Ok(None);
    }
    let mut names = Vec::with_capacity(decoded.key_field_ids.len());
    for (id, key_type) in decoded.key_field_ids.iter().zip(&decoded.key_types) {
        let Some(field) = fields.iter().find(|field| field.id == *id) else {
            return Ok(None);
        };
        if field.logical_type.to_string() != *key_type {
            return Ok(None);
        }
        names.push(field.name.clone());
    }
    decoded.keys = names;
    Ok(Some(prost_types::Any::from_msg(&decoded)?))
}

/// The indexed column of `index` in `schema`.
fn indexed_column<'a>(
    schema: &'a lance_core::datatypes::Schema,
    index: &IndexMetadata,
) -> Option<&'a Field> {
    schema.field_by_id(*index.fields.first()?)
}

/// The details to open a list-element segment with: bound to the current schema
/// when its keys still resolve, so an update reads the right children after a
/// rename, and the stored ones otherwise. A segment that no longer resolves is
/// never queried, but it can still be remapped by compaction.
pub fn details_for_open(
    schema: &lance_core::datatypes::Schema,
    index: &IndexMetadata,
    details: Arc<prost_types::Any>,
) -> Result<Arc<prost_types::Any>> {
    let Some(column) = indexed_column(schema, index) else {
        return Ok(details);
    };
    Ok(match bind_key_fields(column, &details)? {
        Some(bound) => Arc::new(bound),
        None => details,
    })
}

/// Whether the key fields of list-element segment `index` still resolve in
/// `schema`. One that does not is never queried, so it is not worth updating.
pub fn key_fields_resolve(
    schema: &lance_core::datatypes::Schema,
    index: &IndexMetadata,
) -> Result<bool> {
    let (Some(column), Some(details)) = (indexed_column(schema, index), &index.index_details)
    else {
        return Ok(false);
    };
    Ok(bind_key_fields(column, details)?.is_some())
}

/// Whether `index`, possibly cached before a rename, names its keys other than
/// `details` do. Searching does not use the names, but updating does.
pub fn has_other_key_names(index: &dyn ScalarIndex, details: &prost_types::Any) -> bool {
    let Some(index) = index.as_any().downcast_ref::<ListElementIndex>() else {
        return false;
    };
    decode(details).is_ok_and(|details| index.key_names() != details.keys)
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_schema::{DataType, Field as ArrowField, Fields};
    use lance_index::pb::ListElementKeyEncoding;
    use rstest::rstest;

    /// `detections: list<struct<label: string, other: string, confidence: float>>`
    /// with the struct's fields at IDs 2, 3 and 4.
    fn column() -> Field {
        let element = Fields::from(vec![
            ArrowField::new("label", DataType::Utf8, true),
            ArrowField::new("other", DataType::Utf8, true),
            ArrowField::new("confidence", DataType::Float32, true),
        ]);
        let arrow = ArrowField::new(
            "detections",
            DataType::List(Arc::new(ArrowField::new(
                "item",
                DataType::Struct(element),
                true,
            ))),
            true,
        );
        let mut schema =
            lance_core::datatypes::Schema::try_from(&arrow_schema::Schema::new(vec![arrow]))
                .unwrap();
        schema.set_field_id(None);
        let column = schema.field("detections").unwrap().clone();
        let ids = element_fields(&column)
            .unwrap()
            .iter()
            .map(|f| f.id)
            .collect::<Vec<_>>();
        assert_eq!(ids, [2, 3, 4]);
        column
    }

    fn details(ids: &[i32], keys: &[&str], key_types: &[&str]) -> prost_types::Any {
        prost_types::Any::from_msg(&ListElementIndexDetails {
            key_field_ids: ids.to_vec(),
            keys: keys.iter().map(|k| k.to_string()).collect(),
            key_types: key_types.iter().map(|t| t.to_string()).collect(),
            key_encoding: ListElementKeyEncoding::ArrowRowV1.into(),
            target_details: None,
            target_index_version: 0,
        })
        .unwrap()
    }

    fn bound_keys(column: &Field, details: &prost_types::Any) -> Option<Vec<String>> {
        bind_key_fields(column, details)
            .unwrap()
            .map(|bound| decode(&bound).unwrap().keys)
    }

    #[test]
    fn test_binding_follows_ids() {
        let mut column = column();
        let label = details(&[2, 4], &["label", "confidence"], &["string", "float"]);
        assert_eq!(
            bound_keys(&column, &label),
            Some(vec!["label".to_string(), "confidence".to_string()])
        );
        // Swap the names of fields 2 and 3: the index follows field 2.
        let children = &mut column.children[0].children;
        children[0].name = "other".to_string();
        children[1].name = "label".to_string();
        assert_eq!(
            bound_keys(&column, &label),
            Some(vec!["other".to_string(), "confidence".to_string()])
        );
    }

    #[rstest]
    #[case::dropped_field(&[5], &["label"], &["string"])]
    #[case::changed_type(&[4], &["confidence"], &["double"])]
    #[case::ids_not_recorded(&[], &["label"], &["string"])]
    fn test_binding_declines(#[case] ids: &[i32], #[case] keys: &[&str], #[case] types: &[&str]) {
        assert_eq!(bound_keys(&column(), &details(ids, keys, types)), None);
    }

    #[test]
    fn test_record_key_field_ids() {
        let mut created = CreatedIndex {
            index_details: details(&[], &["confidence", "label"], &["float", "string"]),
            index_version: 0,
            files: Vec::new(),
        };
        record_key_field_ids(&column(), &mut created).unwrap();
        assert_eq!(
            decode(&created.index_details).unwrap().key_field_ids,
            [4, 2]
        );
    }
}
