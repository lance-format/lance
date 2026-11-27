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

use std::borrow::Cow;
use std::sync::{Arc, Mutex, Weak};

use arrow_schema::{Field as ArrowField, Schema as ArrowSchema, SchemaRef};
use lance_core::cache::{CacheKey, CacheKeySchema, KeyBuilder};
use lance_core::datatypes::Schema;
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_table::format::Manifest;

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

/// A filter schema in the session cache, with the manifest it was built from.
///
/// A key's version and e_tag need not identify a manifest: a table recreated
/// at the same URI and version can repeat them, or leave the e_tag unset. An
/// entry only answers for the manifest allocation it holds a `Weak` to, which
/// that `Weak` keeps from being reused.
#[derive(Debug)]
pub struct SharedFilterSchema {
    manifest: Weak<Manifest>,
    pub filter: FilterSchema,
}

impl SharedFilterSchema {
    pub fn new(manifest: &Arc<Manifest>, filter: FilterSchema) -> Self {
        Self {
            manifest: Arc::downgrade(manifest),
            filter,
        }
    }

    pub fn is_for(&self, manifest: &Arc<Manifest>) -> bool {
        std::ptr::eq(self.manifest.as_ptr(), Arc::as_ptr(manifest))
    }
}

impl DeepSizeOf for SharedFilterSchema {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.filter.deep_size_of_children(context)
    }
}

/// The filter schema of one manifest version, for one set of search columns.
#[derive(Debug, Clone, Copy)]
pub struct FilterSchemaKey<'a> {
    pub version: u64,
    pub e_tag: Option<&'a str>,
    pub extra_columns: &'a [ArrowField],
}

impl CacheKey for FilterSchemaKey<'_> {
    type ValueType = SharedFilterSchema;
    fn key(&self) -> Cow<'_, str> {
        let extra = self
            .extra_columns
            .iter()
            .map(|f| format!("{}:{}:{}", f.name(), f.data_type(), f.is_nullable()))
            .collect::<Vec<_>>()
            .join(",");
        Cow::Owned(format!(
            "filter_schema/{}/{}/{extra}",
            self.version,
            self.e_tag.unwrap_or_default()
        ))
    }
    fn type_name() -> &'static str {
        "FilterSchema"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.dataset.filter-schema-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_u64(self.version);
        if let Some(e_tag) = self.e_tag {
            builder.write_some();
            builder.write_str(e_tag);
        } else {
            builder.write_none();
        }
        builder.write_sequence_len(self.extra_columns.len() as u64);
        for field in self.extra_columns {
            builder.write_str(field.name());
            builder.write_str(&field.data_type().to_string());
            builder.write_bool(field.is_nullable());
        }
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
