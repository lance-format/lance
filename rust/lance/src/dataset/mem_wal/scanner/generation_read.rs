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
use lance_core::is_system_column;
use lance_core::{Error, Result};

use super::exec::ReconcileExec;
use crate::dataset::mem_wal::reconcile::{Plan, field_id_of, without_field_id};
use crate::dataset::mem_wal::{TOMBSTONE, arrow_schema_with_field_ids};

/// One sealed generation, read under the table's schema.
///
/// Answers what to project ([`Self::stored_projection`]), whether a predicate
/// can be pushed down and under which names ([`Self::to_stored`]), and how to
/// bring the result back to the table's names ([`Self::reconcile`]).
///
/// # Invariant
///
/// A field id must keep identifying the same column while any generation holds
/// it. Lance does not enforce this: `max_field_id` is a maximum over the current
/// schema and base fragments, so dropping a column can lower it and let the next
/// added column reuse the id. Reads then serve the dropped column's values as
/// the new one. Callers that retain generations must drain them before a
/// drop/add can reuse an id.
pub(super) struct GenerationRead {
    /// The generation's own schema, carrying its field ids.
    stored_schema: Schema,
    /// The generation's name for a column → the table's name for it, matched
    /// by field id. Carried as names because everything downstream is
    /// name-addressed: a scan projection takes names, and so do the column
    /// references in a DataFusion predicate. Columns the table no longer has
    /// are absent.
    names: HashMap<String, String>,
    /// The table's schema, also carrying field ids.
    table_schema: SchemaRef,
    pk_columns: Vec<String>,
    /// The columns this read produces, in order, each under the name the
    /// table gives it. Not only what the caller asked for --
    /// [`Self::also_produce`] adds what a deferred predicate needs.
    projection: Vec<String>,
}

impl GenerationRead {
    /// `projection` is what the caller asks for, under the table's names.
    pub(super) fn new(
        dataset_schema: &lance_core::datatypes::Schema,
        table_schema: &SchemaRef,
        pk_columns: &[String],
        projection: Vec<String>,
    ) -> Self {
        let stored_schema = arrow_schema_with_field_ids(dataset_schema);
        let names = stored_names(&stored_schema, table_schema);
        let (table_schema, pk_columns) = (Arc::clone(table_schema), pk_columns.to_vec());
        Self {
            stored_schema,
            names,
            table_schema,
            pk_columns,
            projection,
        }
    }

    /// The generation's name for `column`. `column` is the table's name for it.
    pub(super) fn stored_name(&self, column: &str) -> Option<&str> {
        self.names
            .iter()
            .find(|(_, in_table)| *in_table == column)
            .map(|(in_generation, _)| in_generation.as_str())
    }

    /// Also produce `column`, which the caller needs even though it did not ask
    /// for it — a predicate that runs after reconciliation reads its columns
    /// from this scan.
    pub(super) fn also_produce(&mut self, column: &str) {
        if !self.projection.iter().any(|w| w == column) {
            self.projection.push(column.to_string());
        }
    }

    /// What to project from the file: the projected columns under the names it
    /// has.
    /// A column it never stored is dropped here and filled in by
    /// [`Self::reconcile`].
    pub(super) fn stored_projection(&self) -> Vec<&str> {
        self.projection
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
            .then(|| self.stored_schema.field_with_name(name).ok())
            .flatten()
            .map(|f| f.name().as_str())
    }

    /// `expr` with each column reference moved to the name this generation
    /// stores it under, so it can be pushed into the generation's own scan.
    ///
    /// `None` when the generation does not store a referenced column, or stores
    /// it under a different shape; the predicate then runs above the
    /// reconciliation instead. A nested reference names its parent (`info.a`
    /// refers to `info`), and a child rename does not move the parent's name, so
    /// the parent's whole type is compared rather than its name.
    pub(super) fn to_stored(&self, expr: &Expr) -> Option<Expr> {
        let pushable = expr
            .column_refs()
            .iter()
            .all(|c| self.stored_as_declared(&c.name));
        if !pushable {
            return None;
        }
        expr.clone()
            .transform(|e| match e {
                Expr::Column(mut c) => {
                    // `stored_name` is total over the refs, checked above.
                    c.name = self.stored_name(&c.name).expect("checked").to_string();
                    Ok(Transformed::yes(Expr::Column(c)))
                }
                other => Ok(Transformed::no(other)),
            })
            .map(|t| t.data)
            .ok()
    }

    /// Split `filter` into the part this generation can answer under its own
    /// names and the part that has to run above the reconciliation.
    ///
    /// The deferred part reads its columns from this scan, so they are added to
    /// what the scan produces whether the caller asked for them or not. At most
    /// one of the two is `Some`: a predicate is pushed whole or deferred whole.
    pub(super) fn split_filter(&mut self, filter: Option<&Expr>) -> (Option<Expr>, Option<Expr>) {
        let pushed = filter.and_then(|expr| self.to_stored(expr));
        let deferred = filter.filter(|_| pushed.is_none()).cloned();
        if let Some(expr) = &deferred {
            for column in expr.column_refs() {
                self.also_produce(&column.name);
            }
        }
        (pushed, deferred)
    }

    /// Whether the generation stores `column` exactly as the table declares
    /// it, so a predicate naming it means the same thing pushed down.
    ///
    /// A nested column is where the two can differ without the name moving: a
    /// reference names the parent (`info.a` refers to `info`) and a parent's
    /// name does not move when a child is renamed. Comparing the shapes rather
    /// than assuming the worst is what keeps an ordinary predicate on an
    /// ordinary struct pushed down — the common case, where nothing moved.
    fn stored_as_declared(&self, column: &str) -> bool {
        let Some(stored_column) = self.stored_name(column) else {
            return false;
        };
        let (Ok(stored_field), Ok(declared)) = (
            self.stored_schema.field_with_name(stored_column),
            self.table_schema.field_with_name(column),
        ) else {
            return false;
        };
        // The top-level ids matched already: `stored_name` resolved through a
        // map keyed by them. This compares the children, whose ids live inside
        // the parent's own type -- so a child dropped and added back under the
        // same name and type is caught as the different column it is, rather
        // than filtering on the retired child's values while the reconciliation
        // above synthesizes nulls for the new one.
        stored_field.data_type() == declared.data_type()
    }

    /// Bring the scan's output back to the table's names and shapes: renames
    /// followed, columns the generation never stored filled with nulls, nested
    /// columns rebuilt to the shape the table declares.
    ///
    /// `scan` may produce more than was asked for (`_rowaddr`, `_tombstone`);
    /// those pass through untouched, as does anything the generation has that
    /// the table does not.
    pub(super) fn reconcile(&self, scan: Arc<dyn ExecutionPlan>) -> Result<Arc<dyn ExecutionPlan>> {
        let source =
            self.with_only_table_field_ids(with_ids_from(&scan.schema(), &self.stored_schema));
        let target = self.target(&source);
        let plan = Plan::resolve(&source, &target, &self.pk_columns)?.emitting_plain_schema();
        if plan.is_identity() {
            return Ok(scan);
        }
        Ok(Arc::new(ReconcileExec::new(scan, Arc::new(plan))))
    }

    /// `source` keeping a field id only where the column is one of the
    /// table's, and stripping it everywhere else.
    ///
    /// A generation numbers its own columns in its own schema — `_tombstone`,
    /// and anything it holds that the table has since dropped — so those ids
    /// collide with whatever the table gave those numbers. Left in place, a
    /// column added to the table resolves to whichever of them happens to share
    /// its id.
    fn with_only_table_field_ids(&self, source: Schema) -> Schema {
        let fields: Vec<Field> = source
            .fields()
            .iter()
            .map(|field| match self.names.contains_key(field.name()) {
                true => field.as_ref().clone(),
                false => without_field_id(field),
            })
            .collect();
        Schema::new_with_metadata(fields, source.metadata().clone())
    }

    /// The schema [`Self::reconcile`] produces: the projected columns under the
    /// table's names, then whatever else the scan carries.
    ///
    /// Nullability comes from the source, not the table, because generations
    /// store non-key columns as nullable so a strict table can hold a tombstone.
    /// Each arm's canonical projection restores the table's nullability once
    /// tombstones are dropped.
    fn target(&self, source: &Schema) -> SchemaRef {
        let mut fields: Vec<Field> = self
            .projection
            .iter()
            .filter_map(|name| {
                let declared = self.table_schema.field_with_name(name).ok()?;
                // Absent from the source means synthesized, so nullable.
                let nullable = self
                    .stored_name(name)
                    .and_then(|stored_column| source.field_with_name(stored_column).ok())
                    .is_none_or(|f| f.is_nullable());
                Some(declared.clone().with_nullable(nullable))
            })
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
        .map_err(|e| Error::internal(format!("build a filter schema for `{expr}`: {e}")))?;
    let props = ExecutionProps::new();
    let physical = create_physical_expr(expr, &df_schema, &props)
        .map_err(|e| Error::internal(format!("plan filter `{expr}`: {e}")))?;
    Ok(Arc::new(
        FilterExec::try_new(physical, plan).map_err(|e| Error::internal(format!("filter: {e}")))?,
    ))
}

/// Each column the generation stores, keyed by the name it stores it under,
/// mapped to the table's name for it. Paired by field id, since a rename
/// changes the name and keeps the id.
fn stored_names(stored_schema: &Schema, table_schema: &Schema) -> HashMap<String, String> {
    let by_id: HashMap<i32, &str> = table_schema
        .fields()
        .iter()
        .filter_map(|f| field_id_of(f).map(|id| (id, f.name().as_str())))
        .collect();
    // A caller that supplies no ids leaves only names to match on.
    if by_id.is_empty() {
        return stored_schema
            .fields()
            .iter()
            .filter(|f| f.name() != TOMBSTONE && !is_system_column(f.name()))
            .filter(|f| table_schema.field_with_name(f.name()).is_ok())
            .map(|f| (f.name().clone(), f.name().clone()))
            .collect();
    }
    stored_schema
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
fn with_ids_from(schema: &Schema, stored_schema: &Schema) -> Schema {
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
            (DataType::Map(entries, sorted), DataType::Map(source_entries, _)) => {
                let sorted = *sorted;
                let one: Fields = vec![source_entries.as_ref().clone()].into();
                field.with_data_type(DataType::Map(Arc::new(restore(entries, &one)), sorted))
            }
            _ => field,
        }
    }
    let fields: Vec<Field> = schema
        .fields()
        .iter()
        .map(|field| restore(field, stored_schema.fields()))
        .collect();
    Schema::new_with_metadata(fields, schema.metadata().clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::Fields;
    use datafusion::prelude::{col, lit};
    use lance_core::datatypes::LANCE_FIELD_ID_KEY;
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
    fn generation(
        stored_schema: SchemaRef,
        table_schema: SchemaRef,
        projection: &[&str],
    ) -> GenerationRead {
        let lance = LanceSchema::try_from(stored_schema.as_ref()).expect("a lance schema");
        GenerationRead::new(
            &lance,
            &table_schema,
            &["id".to_string()],
            projection.iter().map(|s| s.to_string()).collect(),
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

    /// A predicate on a struct reaches the stored data only where the whole
    /// struct still matches, child names included.
    #[test]
    fn a_struct_predicate_is_pushed_down_only_when_its_children_did_not_move() {
        // The child as the generation stored it, as the table now declares it,
        // and whether a predicate on the parent may reach the stored data.
        let cases = [
            // Deferring this one was the bug: a search takes its top-k first,
            // so a predicate applied afterwards loses a lower-ranked row that
            // should have won.
            ("nothing moved", ("c", 2), ("c", 2), true),
            ("the child was renamed", ("c", 2), ("d", 2), false),
            // Same name, same type, different field id: a child dropped and
            // added back is a different column wearing the old one's shape, so
            // a predicate on it must not reach the retired values.
            ("the child was replaced", ("c", 2), ("c", 9), false),
        ];
        for (case, stored_child, table_child, pushed_down) in cases {
            let info = |(name, id): (&str, i32)| {
                with_id(
                    "info",
                    DataType::Struct(Fields::from(vec![with_id(name, DataType::Int64, id)])),
                    1,
                )
            };
            let read = generation(
                schema(vec![with_id("id", DataType::Int64, 0), info(stored_child)]),
                schema(vec![with_id("id", DataType::Int64, 0), info(table_child)]),
                &["id", "info"],
            );
            let expr = col("info").is_not_null();
            let got = read.to_stored(&expr);
            assert_eq!(
                got,
                pushed_down.then_some(expr),
                "{case}: the predicate was pushed down when it should not have been, or the reverse"
            );
        }
    }

    #[test]
    fn a_renamed_column_is_asked_for_under_the_name_the_generation_has() {
        assert_eq!(renamed().stored_projection(), vec!["id", "value"]);
        assert_eq!(renamed().stored_name("amount"), Some("value"));
    }

    /// What the [`GenerationRead`] field-id invariant costs when a caller does
    /// not keep it: an id taken back by a later column reads as that column,
    /// and `reconcile_batches` pairs them the same way, so a merge persists it.
    #[test]
    fn a_field_id_taken_back_after_a_drop_is_read_as_the_column_that_took_it() {
        let read = generation(
            // Sealed while id 1 was `retired`.
            schema(vec![
                with_id("id", DataType::Int64, 0),
                with_id("retired", DataType::Int64, 1),
            ]),
            // `retired` was dropped and `added` took its id back.
            schema(vec![
                with_id("id", DataType::Int64, 0),
                with_id("added", DataType::Int64, 1),
            ]),
            &["id", "added"],
        );
        assert_eq!(
            read.stored_name("added"),
            Some("retired"),
            "today the reused id pairs them; a fix makes this `None`, and the \
             generation contributes null for `added` instead"
        );
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

    /// With no ids to match on, the table's own names are the only link — the
    /// behaviour a caller that supplies no identity schema gets.
    #[test]
    fn a_table_without_ids_matches_by_name() {
        let stored_schema = Schema::new(vec![
            with_id("id", DataType::Int64, 0),
            with_id("value", DataType::Int64, 1),
            with_id(TOMBSTONE, DataType::Boolean, 2),
        ]);
        let table_schema = Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new("value", DataType::Int64, true),
        ]);
        let names = stored_names(&stored_schema, &table_schema);
        assert_eq!(names.get("value"), Some(&"value".to_string()));
        assert_eq!(names.get(TOMBSTONE), None, "not one of the table's columns");
    }
}
