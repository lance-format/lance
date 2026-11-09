// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::{Arc, Mutex, Weak};

use arrow_schema::{Field as ArrowField, Schema as ArrowSchema, SchemaRef};
use lance_table::format::Manifest;

use crate::Result;
use crate::datatypes::Schema;

/// The schema available to scan filters, in both forms the planner needs.
#[derive(Clone)]
pub struct FilterableSchema {
    pub schema: Arc<Schema>,
    pub arrow: SchemaRef,
}

/// Upper bound on distinct extra-column sets kept per manifest.
const MAX_ENTRIES: usize = 8;

/// Memoizes the schema available to scan filters, per manifest and set of extra columns.
///
/// Building it projects and copies the whole table schema, which dominates
/// planning a filtered scan of a wide table.
///
/// Entries only answer for the manifest allocation they were built from.
/// The held [`Weak`] keeps that address from being reused, makes [`Arc::get_mut`]
/// fail and makes [`Arc::make_mut`] move the manifest, so a manifest edited in
/// place never matches.
#[derive(Default)]
pub struct FilterableSchemaCache(Mutex<Option<Entries>>);

struct Entries {
    manifest: Weak<Manifest>,
    schemas: Vec<(Vec<ArrowField>, FilterableSchema)>,
}

impl Entries {
    fn is_for(&self, manifest: &Arc<Manifest>) -> bool {
        std::ptr::eq(self.manifest.as_ptr(), Arc::as_ptr(manifest))
    }
}

impl FilterableSchemaCache {
    /// Returns the schema `build` produces for `manifest` and `extra_columns`.
    ///
    /// `build` must depend on nothing else.
    pub fn get_or_try_insert(
        &self,
        manifest: &Arc<Manifest>,
        extra_columns: &[ArrowField],
        build: impl FnOnce() -> Result<Schema>,
    ) -> Result<FilterableSchema> {
        if let Some(entries) = self.0.lock().unwrap().as_ref()
            && entries.is_for(manifest)
            && let Some((_, schema)) = entries
                .schemas
                .iter()
                .find(|(columns, _)| columns.as_slice() == extra_columns)
        {
            return Ok(schema.clone());
        }

        let schema = build()?;
        let schema = FilterableSchema {
            arrow: Arc::new(ArrowSchema::from(&schema)),
            schema: Arc::new(schema),
        };
        let mut guard = self.0.lock().unwrap();
        match guard.as_mut() {
            Some(entries) if entries.is_for(manifest) => {
                if entries.schemas.len() >= MAX_ENTRIES {
                    entries.schemas.clear();
                }
                entries
                    .schemas
                    .push((extra_columns.to_vec(), schema.clone()));
            }
            _ => {
                *guard = Some(Entries {
                    manifest: Arc::downgrade(manifest),
                    schemas: vec![(extra_columns.to_vec(), schema.clone())],
                });
            }
        }
        Ok(schema)
    }
}
