// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The schema a scan's filter resolves against.
//!
//! A filter may name any table column, system column (`_rowid`, …) or search
//! column (`_distance`, …), so planning resolves it against all of them, in
//! both Lance and Arrow form. Building that copies every field of the table,
//! so it is kept in the session's metadata cache, one entry per manifest
//! version and set of search columns, where it counts toward the cache's
//! byte budget like any other entry.

use std::sync::{Arc, Mutex};

use arrow_schema::{Field as ArrowField, Schema as ArrowSchema, SchemaRef};
use lance_core::datatypes::Schema;
use lance_core::deepsize::{Context, DeepSizeOf};

/// The table schema plus system and search columns, in Lance and Arrow form.
#[derive(Debug, Clone)]
pub struct FilterSchema {
    pub schema: Arc<Schema>,
    pub arrow: SchemaRef,
}

impl FilterSchema {
    pub fn new(schema: Schema) -> Self {
        Self {
            arrow: Arc::new(ArrowSchema::from(&schema)),
            schema: Arc::new(schema),
        }
    }
}

impl DeepSizeOf for FilterSchema {
    /// Counts the Lance schema twice, as an estimate of it plus its Arrow form.
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        2 * self.schema.deep_size_of_children(context)
    }
}

/// Filter schemas one scanner has already looked up, by their search columns.
///
/// Lets the synchronous filter paths of a plan reuse what planning loaded
/// from the session cache.
#[derive(Clone, Default)]
pub struct ScannerFilterSchemas(Arc<Mutex<Vec<Entry>>>);

type Entry = (Vec<ArrowField>, FilterSchema);

impl ScannerFilterSchemas {
    pub fn get(&self, extra_columns: &[ArrowField]) -> Option<FilterSchema> {
        let entries = self.0.lock().unwrap();
        entries
            .iter()
            .find(|(extra, _)| extra == extra_columns)
            .map(|(_, found)| found.clone())
    }

    pub fn insert(&self, extra_columns: Vec<ArrowField>, schema: FilterSchema) {
        let mut entries = self.0.lock().unwrap();
        if !entries.iter().any(|(extra, _)| *extra == extra_columns) {
            entries.push((extra_columns, schema));
        }
    }
}

#[cfg(test)]
mod tests {
    use arrow_schema::DataType;

    use super::*;

    #[test]
    fn test_deep_size_counts_the_schema() {
        let arrow = ArrowSchema::new(
            (0..100)
                .map(|i| ArrowField::new(format!("c{i}"), DataType::Int32, true))
                .collect::<Vec<_>>(),
        );
        let filter = FilterSchema::new(Schema::try_from(&arrow).unwrap());
        let schema_bytes = filter.schema.as_ref().deep_size_of();
        assert!(filter.deep_size_of() >= 2 * schema_bytes);
    }
}
