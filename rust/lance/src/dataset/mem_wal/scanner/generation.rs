// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Reading a sealed generation under the table's current schema.
//!
//! A sealed generation is a Lance dataset of its own, written under whatever
//! the table's schema was at the time. The names it stores are therefore its
//! own: a rename since the seal moved the table's name and left the file
//! holding the old one, and only field ids relate the two. Every read path —
//! scan, point lookup, vector search, full-text search — goes through
//! [`GenerationRead`] so they resolve a generation the same way.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef};
use datafusion::common::DFSchema;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::execution::context::ExecutionProps;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::filter::FilterExec;
use datafusion::prelude::Expr;
use datafusion_physical_expr::create_physical_expr;
use lance_core::datatypes::LANCE_FIELD_ID_KEY;
use lance_core::is_system_column;
use lance_core::{Error, Result};

use super::exec::ReconcileExec;
use crate::dataset::mem_wal::reconcile::{Plan, field_id_of};
use crate::dataset::mem_wal::{TOMBSTONE, arrow_schema_with_field_ids};

/// One sealed generation, read under the table's schema.
///
/// Built from the generation's own schema and the table's id-carrying schema.
/// It answers three questions, in the order a scan needs them: what to project
/// from the file ([`Self::stored_projection`]), whether a predicate can be
/// pushed into it and under which names ([`Self::to_stored`]), and how to
/// bring the result back to the table's names and shapes
/// ([`Self::reconcile`]).
pub(super) struct GenerationRead {
    /// The generation's own schema, carrying its field ids.
    stored: Schema,
    /// The generation's name for a column → the table's name for it. Columns
    /// the table no longer has are absent.
    names: HashMap<String, String>,
    /// The table's schema, carrying field ids.
    identity: SchemaRef,
    pk_columns: Vec<String>,
    /// Table names this read produces, in order.
    wanted: Vec<String>,
}

impl GenerationRead {
    /// `wanted` is what the caller asks for, in the table's names.
    pub(super) fn new(
        dataset_schema: &lance_core::datatypes::Schema,
        identity: SchemaRef,
        pk_columns: Vec<String>,
        wanted: Vec<String>,
    ) -> Self {
        let stored = arrow_schema_with_field_ids(dataset_schema);
        let names = stored_names(&stored, &identity);
        Self {
            stored,
            names,
            identity,
            pk_columns,
            wanted,
        }
    }

    /// The generation's name for one of the table's columns.
    pub(super) fn stored_name(&self, table_name: &str) -> Option<&str> {
        self.names
            .iter()
            .find(|(_, table)| *table == table_name)
            .map(|(stored, _)| stored.as_str())
    }

    /// Also produce `column`, which the caller needs even though it did not ask
    /// for it — a predicate that runs after reconciliation reads its columns
    /// from this scan.
    pub(super) fn also_produce(&mut self, column: &str) {
        if !self.wanted.iter().any(|w| w == column) {
            self.wanted.push(column.to_string());
        }
    }

    /// What to project from the file: the wanted columns under the names it has.
    /// A column it never stored is dropped here and filled in by
    /// [`Self::reconcile`].
    pub(super) fn stored_projection(&self) -> Vec<&str> {
        self.wanted
            .iter()
            .filter_map(|name| {
                self.stored_name(name)
                    .or_else(|| self.stored_system_column(name))
            })
            .collect()
    }

    /// A system column (`_tombstone`, `_rowaddr`) is not one of the table's, so
    /// no field id relates it; the generation stores it under the name it is
    /// asked for, or not at all.
    fn stored_system_column(&self, name: &str) -> Option<&str> {
        (is_system_column(name) || name == TOMBSTONE)
            .then(|| self.stored.field_with_name(name).ok())
            .flatten()
            .map(|f| f.name().as_str())
    }

    /// `expr` with each column reference moved to the name this generation
    /// stores it under, so it can be pushed into the generation's own scan.
    ///
    /// `None` when it cannot be: a column the generation does not store, or a
    /// nested one. A nested reference names the parent (`info.a` refers to
    /// `info`) and a parent's name does not move when a child is renamed, so a
    /// pushed-down predicate would be evaluated against child names this
    /// generation has and the table does not. Either way the predicate belongs
    /// above the reconciliation, where the columns it names exist.
    pub(super) fn to_stored(&self, expr: &Expr) -> Option<Expr> {
        let pushable = expr.column_refs().iter().all(|c| {
            self.stored_name(&c.name).is_some_and(|stored| {
                self.stored
                    .field_with_name(stored)
                    .is_ok_and(|f| !is_reconstructed(f.data_type()))
            })
        });
        if !pushable {
            return None;
        }
        expr.clone()
            .transform(|e| match e {
                Expr::Column(mut c) => {
                    // `stored_name` is total over the refs, checked above.
                    let stored = self.stored_name(&c.name).expect("checked").to_string();
                    c.name = stored;
                    Ok(Transformed::yes(Expr::Column(c)))
                }
                other => Ok(Transformed::no(other)),
            })
            .map(|t| t.data)
            .ok()
    }

    /// Bring the scan's output back to the table's names and shapes: renames
    /// followed, columns the generation never stored filled with nulls, nested
    /// columns rebuilt to the shape the table declares.
    ///
    /// `scan` may produce more than was asked for (`_rowaddr`, `_tombstone`);
    /// those pass through untouched, as does anything the generation has that
    /// the table does not.
    pub(super) fn reconcile(&self, scan: Arc<dyn ExecutionPlan>) -> Result<Arc<dyn ExecutionPlan>> {
        let source = self.only_the_tables_ids(with_ids_from(&scan.schema(), &self.stored));
        let target = self.target(&source);
        let plan = Plan::resolve(&source, &target, &self.pk_columns)?.emitting_plain_schema();
        if plan.is_identity() {
            return Ok(scan);
        }
        Ok(Arc::new(ReconcileExec::new(scan, Arc::new(plan))))
    }

    /// `source` with the field ids of everything that is not one of the
    /// table's columns removed.
    ///
    /// A generation numbers its own columns in its own schema — `_tombstone`,
    /// and anything it holds that the table has since dropped — so those ids
    /// collide with whatever the table gave those numbers. Left in place, a
    /// column added to the table resolves to whichever of them happens to share
    /// its id.
    fn only_the_tables_ids(&self, source: Schema) -> Schema {
        let fields: Vec<Field> = source
            .fields()
            .iter()
            .map(|field| match self.names.contains_key(field.name()) {
                true => field.as_ref().clone(),
                false => {
                    let mut metadata = field.metadata().clone();
                    metadata.remove(LANCE_FIELD_ID_KEY);
                    field.as_ref().clone().with_metadata(metadata)
                }
            })
            .collect();
        Schema::new_with_metadata(fields, source.metadata().clone())
    }

    /// The schema [`Self::reconcile`] produces: the wanted columns as the table
    /// declares them, then whatever else the scan carries.
    fn target(&self, source: &Schema) -> SchemaRef {
        let mut fields: Vec<Field> = self
            .wanted
            .iter()
            .filter_map(|name| self.identity.field_with_name(name).ok().cloned())
            .collect();
        // A generation's own columns are not the table's, so they pass through
        // as the generation has them.
        for field in source.fields() {
            let is_the_tables = self.names.contains_key(field.name());
            if !is_the_tables && fields.iter().all(|f| f.name() != field.name()) {
                fields.push(field.as_ref().clone());
            }
        }
        Arc::new(Schema::new(fields))
    }
}

/// Run `expr` above `plan`, for a predicate that could not be pushed into the
/// generation's own scan.
pub(super) fn filter_above(
    plan: Arc<dyn ExecutionPlan>,
    expr: &Expr,
) -> Result<Arc<dyn ExecutionPlan>> {
    let schema = plan.schema();
    let df_schema = DFSchema::try_from(schema.as_ref().clone())
        .map_err(|e| Error::internal(format!("filter schema: {e}")))?;
    let props = ExecutionProps::new();
    let physical = create_physical_expr(expr, &df_schema, &props)
        .map_err(|e| Error::internal(format!("plan filter `{expr}`: {e}")))?;
    Ok(Arc::new(
        FilterExec::try_new(physical, plan).map_err(|e| Error::internal(format!("filter: {e}")))?,
    ))
}

/// A column the reconciliation rebuilds rather than takes as it stands, so its
/// stored shape is not the shape a pushed-down predicate would expect.
fn is_reconstructed(data_type: &DataType) -> bool {
    match data_type {
        DataType::Struct(_) => true,
        DataType::List(e) | DataType::LargeList(e) | DataType::FixedSizeList(e, _) => {
            is_reconstructed(e.data_type())
        }
        _ => false,
    }
}

/// The generation's name for each of the table's columns, by field id: a rename
/// changes the name and keeps the id.
pub(super) fn stored_names(stored: &Schema, table: &Schema) -> HashMap<String, String> {
    let by_id: HashMap<i32, &str> = table
        .fields()
        .iter()
        .filter_map(|f| field_id_of(f).map(|id| (id, f.name().as_str())))
        .collect();
    // A caller that supplies no ids leaves only names to match on.
    if by_id.is_empty() {
        return stored
            .fields()
            .iter()
            .filter(|f| f.name() != TOMBSTONE && !is_system_column(f.name()))
            .filter(|f| table.field_with_name(f.name()).is_ok())
            .map(|f| (f.name().clone(), f.name().clone()))
            .collect();
    }
    stored
        .fields()
        .iter()
        // A generation's own columns are numbered in its own schema, so their
        // ids collide with whatever the table gave those numbers. They are not
        // the table's columns and are never resolved to one.
        .filter(|f| f.name() != TOMBSTONE && !is_system_column(f.name()))
        .filter_map(|f| {
            let id = field_id_of(f)?;
            by_id
                .get(&id)
                .map(|name| (f.name().clone(), name.to_string()))
        })
        .collect()
}

/// Put back the field ids a scan's output schema drops, so the reconciliation
/// can resolve its columns by id.
fn with_ids_from(schema: &Schema, stored: &Schema) -> Schema {
    fn restore(field: &Field, among: &Fields) -> Field {
        let Some(source) = among.iter().find(|f| f.name() == field.name()) else {
            return field.clone();
        };
        let mut metadata = field.metadata().clone();
        metadata.extend(source.metadata().clone());
        let field = field.clone().with_metadata(metadata);
        // A struct's children carry their own ids, and a child can be renamed
        // while its parent's name does not move.
        let data_type = field.data_type().clone();
        match (&data_type, source.data_type()) {
            (DataType::Struct(children), DataType::Struct(source_children)) => {
                let children: Vec<Field> = children
                    .iter()
                    .map(|child| restore(child, source_children))
                    .collect();
                field.with_data_type(DataType::Struct(children.into()))
            }
            // A list's element is a field with an id of its own, and so are its
            // children in turn.
            (DataType::List(element), DataType::List(source_element)) => {
                let one: Fields = vec![source_element.as_ref().clone()].into();
                field.with_data_type(DataType::List(Arc::new(restore(element, &one))))
            }
            (DataType::LargeList(element), DataType::LargeList(source_element)) => {
                let one: Fields = vec![source_element.as_ref().clone()].into();
                field.with_data_type(DataType::LargeList(Arc::new(restore(element, &one))))
            }
            (
                DataType::FixedSizeList(element, size),
                DataType::FixedSizeList(source_element, _),
            ) => {
                let size = *size;
                let one: Fields = vec![source_element.as_ref().clone()].into();
                field.with_data_type(DataType::FixedSizeList(
                    Arc::new(restore(element, &one)),
                    size,
                ))
            }
            _ => field,
        }
    }
    let fields: Vec<Field> = schema
        .fields()
        .iter()
        .map(|field| restore(field, stored.fields()))
        .collect();
    Schema::new_with_metadata(fields, schema.metadata().clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::Fields;
    use datafusion::prelude::{col, lit};
    use lance_core::datatypes::Schema as LanceSchema;

    /// An Arrow field carrying a Lance field id, as a generation's schema and
    /// the table's both do.
    fn with_id(name: &str, data_type: DataType, id: i32) -> Field {
        Field::new(name, data_type, true).with_metadata(
            [(LANCE_FIELD_ID_KEY.to_string(), id.to_string())]
                .into_iter()
                .collect(),
        )
    }

    fn schema(fields: Vec<Field>) -> SchemaRef {
        Arc::new(Schema::new(fields))
    }

    /// `GenerationRead` resolves against a generation's *Lance* schema, which
    /// is where the stored ids come from.
    fn generation(stored: SchemaRef, table: SchemaRef, wanted: &[&str]) -> GenerationRead {
        let lance = LanceSchema::try_from(stored.as_ref()).expect("a lance schema");
        GenerationRead::new(
            &lance,
            table,
            vec!["id".to_string()],
            wanted.iter().map(|s| s.to_string()).collect(),
        )
    }

    /// The generation was sealed as `value`; the table has since renamed it.
    fn renamed() -> GenerationRead {
        generation(
            schema(vec![
                with_id("id", DataType::Int64, 0),
                with_id("value", DataType::Int64, 1),
            ]),
            schema(vec![
                with_id("id", DataType::Int64, 0),
                with_id("amount", DataType::Int64, 1),
            ]),
            &["id", "amount"],
        )
    }

    #[test]
    fn a_renamed_column_is_asked_for_under_the_name_the_generation_has() {
        assert_eq!(renamed().stored_projection(), vec!["id", "value"]);
        assert_eq!(renamed().stored_name("amount"), Some("value"));
    }

    #[test]
    fn a_column_the_generation_never_stored_is_left_out_of_the_projection() {
        let read = generation(
            schema(vec![with_id("id", DataType::Int64, 0)]),
            schema(vec![
                with_id("id", DataType::Int64, 0),
                with_id("added", DataType::Int64, 7),
            ]),
            &["id", "added"],
        );
        assert_eq!(read.stored_projection(), vec!["id"]);
        assert_eq!(read.stored_name("added"), None);
    }

    /// A generation numbers its own columns in its own schema, so `_tombstone`
    /// carries an id that collides with whatever the table gave that number.
    #[test]
    fn a_system_column_never_answers_for_one_of_the_tables() {
        let read = generation(
            schema(vec![
                with_id("id", DataType::Int64, 0),
                with_id(TOMBSTONE, DataType::Boolean, 7),
            ]),
            schema(vec![
                with_id("id", DataType::Int64, 0),
                with_id("added", DataType::Int64, 7),
            ]),
            &["id", "added", TOMBSTONE],
        );
        assert_eq!(read.stored_name("added"), None, "not the tombstone's id");
        assert_eq!(
            read.stored_projection(),
            vec!["id", TOMBSTONE],
            "the tombstone is still asked for, under its own name"
        );
    }

    #[test]
    fn a_predicate_naming_a_renamed_column_is_rewritten_to_the_stored_name() {
        let read = renamed();
        assert_eq!(
            read.to_stored(&col("amount").eq(lit(1i64))),
            Some(col("value").eq(lit(1i64))),
        );
    }

    #[test]
    fn a_predicate_naming_a_column_that_did_not_move_is_left_alone() {
        let read = renamed();
        let expr = col("id").eq(lit(1i64));
        assert_eq!(read.to_stored(&expr), Some(expr));
    }

    /// A predicate on a column the generation never stored cannot be pushed
    /// down; it belongs above the reconciliation, where the column exists as
    /// nulls.
    #[test]
    fn a_predicate_naming_a_column_the_generation_lacks_is_not_pushable() {
        let read = generation(
            schema(vec![with_id("id", DataType::Int64, 0)]),
            schema(vec![
                with_id("id", DataType::Int64, 0),
                with_id("added", DataType::Int64, 7),
            ]),
            &["id", "added"],
        );
        assert_eq!(read.to_stored(&col("added").eq(lit(1i64))), None);
    }

    /// A nested reference names the parent, and a parent's name does not move
    /// when a child is renamed — so pushing it down would evaluate it against
    /// child names the table does not have.
    #[test]
    fn a_predicate_on_a_nested_column_is_never_pushed_down() {
        let nested = |child: &str| {
            with_id(
                "info",
                DataType::Struct(Fields::from(vec![with_id(child, DataType::Int64, 2)])),
                1,
            )
        };
        let read = generation(
            schema(vec![with_id("id", DataType::Int64, 0), nested("c")]),
            schema(vec![with_id("id", DataType::Int64, 0), nested("d")]),
            &["id", "info"],
        );
        assert_eq!(read.to_stored(&col("info").is_not_null()), None);
    }

    /// A list of structs is rebuilt too, so its stored shape is not the shape a
    /// pushed-down predicate would expect.
    #[test]
    fn a_list_is_reconstructed_only_when_its_element_is() {
        let struct_element =
            DataType::Struct(Fields::from(vec![Field::new("c", DataType::Int64, true)]));
        assert!(is_reconstructed(&DataType::List(Arc::new(Field::new(
            "item",
            struct_element,
            true
        )))));
        assert!(!is_reconstructed(&DataType::List(Arc::new(Field::new(
            "item",
            DataType::Int64,
            true
        )))));
    }

    /// With no ids to match on, the table's own names are the only link — the
    /// behaviour a caller that supplies no identity schema gets.
    #[test]
    fn a_table_without_ids_matches_by_name() {
        let stored = Schema::new(vec![
            with_id("id", DataType::Int64, 0),
            with_id("value", DataType::Int64, 1),
            with_id(TOMBSTONE, DataType::Boolean, 2),
        ]);
        let table = Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new("value", DataType::Int64, true),
        ]);
        let names = stored_names(&stored, &table);
        assert_eq!(names.get("value"), Some(&"value".to_string()));
        assert_eq!(names.get(TOMBSTONE), None, "not one of the table's columns");
    }
}
