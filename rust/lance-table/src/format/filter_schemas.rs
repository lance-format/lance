// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Schemas that scan filters resolve against, derived from a [`Manifest`]'s
//! schema.
//!
//! A filter may name any table column, system column (`_rowid`, …) or search
//! column (`_distance`, …), so planning resolves it against all of them, in
//! both Lance and Arrow form. Building that copies every field of the table,
//! so a manifest keeps the few variants it is asked for.
//!
//! The entries describe the manifest's `schema` as it was when first asked
//! for. A clone of the manifest starts empty, so changing the schema through
//! `Arc::make_mut` on a shared manifest is safe; editing the schema of a
//! manifest that is already in use is not.
//!
//! [`Manifest`]: super::Manifest

use std::sync::{Arc, Mutex};

use arrow_schema::{Field as ArrowField, Schema as ArrowSchema, SchemaRef};
use lance_core::Result;
use lance_core::datatypes::Schema;
use lance_core::deepsize::{Context, DeepSizeOf};

/// Bounds the variants kept per manifest; scans use a handful of distinct
/// sets of search columns.
const MAX_ENTRIES: usize = 8;

/// The schema available to a scan's filter, in Lance and Arrow form.
#[derive(Debug, Clone)]
pub struct FilterSchema {
    pub schema: Arc<Schema>,
    pub arrow: SchemaRef,
}

/// Filter schemas of one manifest, by the search columns they include.
#[derive(Default)]
pub struct FilterSchemas(Mutex<Vec<(Vec<ArrowField>, FilterSchema)>>);

impl FilterSchemas {
    /// Returns the schema for `extra_columns`, building it with `build` the
    /// first time.
    pub fn get_or_try_insert(
        &self,
        extra_columns: &[ArrowField],
        build: impl FnOnce() -> Result<Schema>,
    ) -> Result<FilterSchema> {
        if let Some(found) = self.find(extra_columns) {
            return Ok(found);
        }
        let schema = build()?;
        let built = FilterSchema {
            arrow: Arc::new(ArrowSchema::from(&schema)),
            schema: Arc::new(schema),
        };
        let mut entries = self.0.lock().unwrap();
        if let Some((_, found)) = entries.iter().find(|(extra, _)| extra == extra_columns) {
            return Ok(found.clone());
        }
        if entries.len() < MAX_ENTRIES {
            entries.push((extra_columns.to_vec(), built.clone()));
        }
        Ok(built)
    }

    fn find(&self, extra_columns: &[ArrowField]) -> Option<FilterSchema> {
        let entries = self.0.lock().unwrap();
        entries
            .iter()
            .find(|(extra, _)| extra == extra_columns)
            .map(|(_, found)| found.clone())
    }
}

/// Starts empty: the clone may have its schema changed before it is used.
impl Clone for FilterSchemas {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl std::fmt::Debug for FilterSchemas {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FilterSchemas")
            .field("entries", &self.0.lock().unwrap().len())
            .finish()
    }
}

/// Derived state: two manifests with the same content are equal whatever
/// either has cached.
impl PartialEq for FilterSchemas {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl DeepSizeOf for FilterSchemas {
    /// Counts each Lance schema twice, as an estimate of it plus its Arrow form.
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        let entries = self.0.lock().unwrap();
        entries
            .iter()
            .map(|(_, found)| {
                if context.mark_seen(Arc::as_ptr(&found.schema) as usize) {
                    2 * found.schema.deep_size_of_children(context)
                } else {
                    0
                }
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use arrow_schema::DataType;

    use super::*;

    fn table() -> Schema {
        Schema::try_from(&ArrowSchema::new(vec![ArrowField::new(
            "a",
            DataType::Int32,
            true,
        )]))
        .unwrap()
    }

    #[test]
    fn test_builds_once_per_set_of_extra_columns() {
        let schemas = FilterSchemas::default();
        let distance = vec![ArrowField::new("_distance", DataType::Float32, true)];
        let mut builds = 0;
        let mut build = || {
            builds += 1;
            Ok(table())
        };
        let first = schemas.get_or_try_insert(&[], &mut build).unwrap();
        let again = schemas.get_or_try_insert(&[], &mut build).unwrap();
        let other = schemas.get_or_try_insert(&distance, &mut build).unwrap();
        assert!(Arc::ptr_eq(&first.schema, &again.schema));
        assert!(!Arc::ptr_eq(&first.schema, &other.schema));
        assert_eq!(*first.arrow, ArrowSchema::from(first.schema.as_ref()));
        assert_eq!(builds, 2);
    }

    #[test]
    fn test_clone_starts_empty() {
        let schemas = FilterSchemas::default();
        let first = schemas.get_or_try_insert(&[], || Ok(table())).unwrap();
        let cloned = schemas.clone();
        let rebuilt = cloned.get_or_try_insert(&[], || Ok(table())).unwrap();
        assert!(!Arc::ptr_eq(&first.schema, &rebuilt.schema));
        let kept = schemas.get_or_try_insert(&[], || Ok(table())).unwrap();
        assert!(Arc::ptr_eq(&first.schema, &kept.schema));
    }
}
