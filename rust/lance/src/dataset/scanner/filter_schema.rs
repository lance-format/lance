// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The schema a scan's filter resolves against.
//!
//! A filter may name any table column, system column (`_rowid`, …) or search
//! column (`_distance`, …). The table and system columns are converted to
//! Arrow once per manifest and kept on the dataset; a scan appends its search
//! columns, which only clones the fields' `Arc`s.

use std::sync::{Arc, Mutex};

use arrow_schema::{Field as ArrowField, Fields, Schema as ArrowSchema, SchemaRef};
use lance_core::datatypes::{Projection, Schema};
use lance_table::format::Manifest;

use crate::Dataset;
use crate::Result;

/// A dataset's table schema plus system columns, in Arrow form, built on
/// first use.
///
/// [`Dataset::set_manifest`] replaces the slot. The slot also remembers the
/// manifest it was built from, since `Dataset::manifest` can be assigned
/// directly.
#[derive(Default)]
pub struct TableFilterSchema(Mutex<Option<(Arc<Manifest>, SchemaRef)>>);

impl std::fmt::Debug for TableFilterSchema {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TableFilterSchema").finish_non_exhaustive()
    }
}

impl TableFilterSchema {
    fn get_or_build(
        &self,
        manifest: &Arc<Manifest>,
        build: impl FnOnce() -> SchemaRef,
    ) -> SchemaRef {
        let mut slot = self.0.lock().unwrap();
        match slot.as_ref() {
            Some((built_for, schema)) if Arc::ptr_eq(built_for, manifest) => schema.clone(),
            _ => {
                let schema = build();
                *slot = Some((manifest.clone(), schema.clone()));
                schema
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn is_built(&self) -> bool {
        self.0.lock().unwrap().is_some()
    }
}

impl Dataset {
    /// The schema a filter resolves against: the table and system columns
    /// plus `extra_columns`.
    pub(crate) fn filterable_schema(
        self: &Arc<Self>,
        extra_columns: &[ArrowField],
    ) -> Result<SchemaRef> {
        let table = self.filter_schema.get_or_build(&self.manifest, || {
            Arc::new(ArrowSchema::from(
                &Projection::full(self.clone())
                    .with_row_id()
                    .with_row_addr()
                    .with_row_last_updated_at_version()
                    .with_row_created_at_version()
                    .to_schema(),
            ))
        });
        if extra_columns
            .iter()
            .any(|extra| table.field_with_name(extra.name()).is_ok())
        {
            // A name the table already has merges into its field.
            let merged = Schema::try_from(table.as_ref())?
                .merge(&ArrowSchema::new(extra_columns.to_vec()))?;
            return Ok(Arc::new(ArrowSchema::from(&merged)));
        }
        let fields = table
            .fields()
            .iter()
            .cloned()
            .chain(extra_columns.iter().cloned().map(Arc::new))
            .collect::<Fields>();
        Ok(Arc::new(ArrowSchema::new_with_metadata(
            fields,
            table.metadata().clone(),
        )))
    }
}
