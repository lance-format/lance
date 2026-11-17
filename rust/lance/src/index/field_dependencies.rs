// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Separate a plugin's query inputs from the full persisted field declaration.
//!
//! Transactions use the persisted declaration without consulting plugins. Only
//! operations that route index input columns use the resolver below.

use std::borrow::Cow;
use std::collections::HashSet;

use lance_core::datatypes::Schema;
use lance_index::scalar::registry::ScalarIndexPlugin;
use lance_table::format::IndexMetadata;

use super::scalar::SCALAR_INDEX_PLUGIN_REGISTRY;
use crate::{Error, Result};

/// Query inputs in their declared order, excluding carried and additional fields.
/// Unknown plugins and legacy details keep their complete keyed declaration;
/// neither is evidence that a multi-field declaration has just one input.
pub fn input_field_ids(index: &IndexMetadata) -> Result<Cow<'_, [i32]>> {
    input_field_ids_for(
        &index.fields,
        &index.covering_fields,
        index.index_details.as_deref(),
    )
}

/// Resolve a staged segment's declaration using the same rules as its metadata.
pub fn input_field_ids_for<'a>(
    fields: &'a [i32],
    covering_fields: &[i32],
    details: Option<&prost_types::Any>,
) -> Result<Cow<'a, [i32]>> {
    let plugin = details.and_then(|details| {
        SCALAR_INDEX_PLUGIN_REGISTRY
            .get_plugin_by_details(details)
            .ok()
    });
    resolve_input_field_ids(fields, covering_fields, details, plugin)
}

fn resolve_input_field_ids<'a>(
    fields: &'a [i32],
    covering_fields: &[i32],
    details: Option<&prost_types::Any>,
    plugin: Option<&dyn ScalarIndexPlugin>,
) -> Result<Cow<'a, [i32]>> {
    let keyed = IndexMetadata::keyed_fields_of(fields, covering_fields);
    let (Some(details), Some(plugin)) = (details, plugin) else {
        return Ok(Cow::Borrowed(keyed));
    };
    let additional = plugin.additional_field_ids(details)?;
    if additional.is_empty() {
        return Ok(Cow::Borrowed(keyed));
    }
    validate_additional_fields(fields, &additional)?;
    if let Some(id) = additional.iter().find(|id| !keyed.contains(id)) {
        return Err(Error::invalid_input(format!(
            "plugin '{}' declares additional field {id} outside its indexed fields {keyed:?}",
            plugin.name(),
        )));
    }
    Ok(Cow::Owned(
        keyed
            .iter()
            .copied()
            .filter(|id| !additional.contains(id))
            .collect(),
    ))
}

/// Persist additional dependencies after the input columns for a newly built index.
/// The plugin must already have recorded any dataset field IDs in its details.
pub fn fields_with_dependencies(
    schema: &Schema,
    inputs: &[i32],
    details: &prost_types::Any,
) -> Result<Vec<i32>> {
    let additional = match SCALAR_INDEX_PLUGIN_REGISTRY.get_plugin_by_details(details) {
        Ok(plugin) => plugin.additional_field_ids(details)?,
        // Vector extensions and unknown types retain their existing declaration.
        Err(_) => Vec::new(),
    };
    let mut fields = Vec::with_capacity(inputs.len() + additional.len());
    fields.extend_from_slice(inputs);
    fields.extend_from_slice(&additional);
    if !additional.is_empty() {
        validate_additional_fields(&fields, &additional)?;
        for id in &additional {
            if schema.field_by_id(*id).is_none() {
                return Err(Error::invalid_input(format!(
                    "index additional field {id} does not exist in the dataset schema"
                )));
            }
        }
    }
    Ok(fields)
}

fn validate_additional_fields(fields: &[i32], additional: &[i32]) -> Result<()> {
    let mut seen = HashSet::with_capacity(fields.len());
    if let Some(id) = fields.iter().find(|id| !seen.insert(**id)) {
        return Err(Error::invalid_input(format!(
            "index field declaration {fields:?} repeats field {id}"
        )));
    }
    let mut seen = HashSet::with_capacity(additional.len());
    for id in additional {
        if *id < 0 || !seen.insert(*id) {
            return Err(Error::invalid_input(format!(
                "index additional fields {additional:?} must be unique user field IDs; invalid field {id}"
            )));
        }
        if !fields.contains(id) {
            return Err(Error::invalid_input(format!(
                "index additional field {id} is absent from its field declaration {fields:?}"
            )));
        }
    }
    if fields.first().is_none_or(|id| additional.contains(id)) {
        return Err(Error::invalid_input(format!(
            "index additional fields {additional:?} consume the primary input of {fields:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;
    use lance_core::cache::LanceCache;
    use lance_index::scalar::expression::ScalarQueryParser;
    use lance_index::scalar::{IndexStore, RowIdRemapper, ScalarIndex};
    use rstest::rstest;

    use super::*;

    #[derive(Debug)]
    struct DependencyPlugin(Vec<i32>);

    #[async_trait]
    impl ScalarIndexPlugin for DependencyPlugin {
        fn name(&self) -> &str {
            "ExternalDependencies"
        }

        fn provides_exact_answer(&self) -> bool {
            true
        }

        fn version(&self) -> u32 {
            0
        }

        fn additional_field_ids(&self, _: &prost_types::Any) -> Result<Vec<i32>> {
            Ok(self.0.clone())
        }

        fn new_query_parser(
            &self,
            _: String,
            _: &prost_types::Any,
        ) -> Option<Box<dyn ScalarQueryParser>> {
            None
        }

        async fn load_index(
            &self,
            _: Arc<dyn IndexStore>,
            _: &prost_types::Any,
            _: u32,
            _: Option<Arc<dyn RowIdRemapper>>,
            _: &LanceCache,
        ) -> Result<Arc<dyn ScalarIndex>> {
            Err(Error::not_supported(
                "the dependency test plugin has no index files",
            ))
        }
    }

    fn details() -> prost_types::Any {
        prost_types::Any {
            type_url: "/lance.test.ExternalDependencies".to_string(),
            value: Vec::new(),
        }
    }

    #[test]
    fn test_plugin_field_dependencies_without_builtin_type_checks() {
        let details = details();
        let plugin = DependencyPlugin(vec![4, 2]);
        let fields = [0, 4, 2];
        assert_eq!(
            resolve_input_field_ids(&fields, &[], Some(&details), Some(&plugin))
                .unwrap()
                .as_ref(),
            [0],
        );
        // Without the external plugin, preserve the entire declaration.
        assert_eq!(
            input_field_ids_for(&fields, &[], Some(&details))
                .unwrap()
                .as_ref(),
            fields,
        );
        assert_eq!(fields, [0, 4, 2]);
    }

    #[test]
    fn test_composite_inputs_are_not_collapsed_by_ancestry() {
        let schema = Schema::try_from(&arrow_schema::Schema::new(vec![arrow_schema::Field::new(
            "parent",
            arrow_schema::DataType::Struct(
                vec![arrow_schema::Field::new(
                    "child",
                    arrow_schema::DataType::Int32,
                    true,
                )]
                .into(),
            ),
            true,
        )]))
        .unwrap();
        let mut schema = schema;
        schema.set_field_id(None);
        let parent = schema.field("parent").unwrap();
        let fields = [parent.id, parent.children[0].id];
        let plugin = DependencyPlugin(Vec::new());
        assert_eq!(
            resolve_input_field_ids(&fields, &[], Some(&details()), Some(&plugin))
                .unwrap()
                .as_ref(),
            fields,
        );
    }

    #[test]
    fn test_legacy_fields_are_preserved_without_details() {
        let fields = [0, 2];
        assert_eq!(
            input_field_ids_for(&fields, &[], None).unwrap().as_ref(),
            fields
        );
    }

    #[test]
    fn test_creation_checks_dependency_schema_presence() {
        let mut schema = Schema::try_from(&arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("input", arrow_schema::DataType::Int32, true),
            arrow_schema::Field::new("dependency", arrow_schema::DataType::Int32, true),
        ]))
        .unwrap();
        schema.set_field_id(None);
        let input = schema.field("input").unwrap().id;
        let dependency = schema.field("dependency").unwrap().id;
        let details = |id| {
            prost_types::Any::from_msg(&lance_index::pb::ListElementIndexDetails {
                key_field_ids: vec![id],
                keys: vec!["dependency".to_string()],
                ..Default::default()
            })
            .unwrap()
        };
        assert_eq!(
            fields_with_dependencies(&schema, &[input], &details(dependency)).unwrap(),
            [input, dependency],
        );
        let error = fields_with_dependencies(&schema, &[input], &details(999)).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains("999 does not exist"), "{error}");
    }

    #[rstest]
    #[case::suffix_covering(&[0, 2, 9], &[9])]
    #[case::independent_covering(&[0, 2], &[9])]
    fn test_resolver_preserves_covering_semantics(
        #[case] fields: &[i32],
        #[case] covering: &[i32],
    ) {
        let plugin = DependencyPlugin(vec![2]);
        assert_eq!(
            resolve_input_field_ids(fields, covering, Some(&details()), Some(&plugin))
                .unwrap()
                .as_ref(),
            [0],
        );
    }

    #[rstest]
    #[case::missing_dependency(vec![0], vec![2])]
    #[case::duplicate_dependency(vec![0, 2], vec![2, 2])]
    #[case::duplicate_persisted_field(vec![0, 2, 2], vec![2])]
    #[case::primary_input(vec![0, 2], vec![0])]
    #[case::negative_dependency(vec![0, -1], vec![-1])]
    fn test_resolver_rejects_malformed_declarations(
        #[case] fields: Vec<i32>,
        #[case] additional: Vec<i32>,
    ) {
        let plugin = DependencyPlugin(additional);
        let error =
            resolve_input_field_ids(&fields, &[], Some(&details()), Some(&plugin)).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains("field"), "{error}");
    }
}
