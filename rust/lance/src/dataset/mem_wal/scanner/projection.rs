// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Projection helpers shared by the LSM vector search, point lookup, and
//! scan planners.
//!
//! `MemTableScanner::project()` only special-cases `_rowid`; passing other
//! system columns through it errors. And cross-LSM values for system
//! columns aren't comparable (a `_rowid` of 5 in the base and in an
//! SSTable refer to different rows).
//!
//! - [`build_scanner_projection`] — strips system / `_distance` cols, appends PKs.
//! - [`canonical_output_schema`] — final schema honoring user order; system
//!   cols become nullable `UInt64`, `_distance` becomes nullable `Float32`.
//! - [`project_to_canonical`] — wraps a plan to emit `target_schema`,
//!   NULL-filling system / `_distance` cols missing from the source.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::{Column, Literal};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::scalar::ScalarValue;
use lance_core::datatypes::{Schema as LanceSchema, parse_field_path};
use lance_core::{ROW_ADDR, ROW_ID, Result, is_system_column};

use super::exec::SchemaRelabelExec;

/// Column name for distance in vector search results.
pub const DISTANCE_COLUMN: &str = "_distance";

/// Did the caller list `_rowid` in their projection?
pub fn wants_row_id(projection: Option<&[String]>) -> bool {
    projection
        .map(|p| p.iter().any(|c| c == ROW_ID))
        .unwrap_or(false)
}

/// Did the caller list `_rowaddr` in their projection?
pub fn wants_row_address(projection: Option<&[String]>) -> bool {
    projection
        .map(|p| p.iter().any(|c| c == ROW_ADDR))
        .unwrap_or(false)
}

/// Auto-managed by the planner; must never reach `scanner.project()`.
fn is_auto_managed(col: &str) -> bool {
    col == DISTANCE_COLUMN || is_system_column(col)
}

/// Resolve projection names — dotted paths included — into the narrowed Arrow
/// fields they select.
///
/// [`LanceSchema::project`] does the two things a flat name lookup cannot: it
/// narrows a struct to the selected leaf (`meta.a` -> `meta: Struct<a>`) and
/// merges sibling leaves of one parent into a single field (`meta.a` +
/// `meta.c` -> `meta: Struct<a, c>`). Fields come back in first-mention order
/// of their top-level column.
fn resolve_data_fields(data_names: &[String], base_schema: &SchemaRef) -> Result<Vec<Arc<Field>>> {
    if data_names.is_empty() {
        return Ok(Vec::new());
    }
    let lance_schema = LanceSchema::try_from(base_schema.as_ref())?;
    let projected = lance_schema.project(data_names)?;
    let arrow_schema = Schema::from(&projected);
    Ok(arrow_schema.fields().iter().cloned().collect())
}

/// Top-level column a (possibly dotted) projection name addresses.
fn top_level_of(name: &str) -> Result<String> {
    parse_field_path(name)?
        .into_iter()
        .next()
        .ok_or_else(|| lance_core::Error::invalid_input(format!("empty projection name: {}", name)))
}

/// Projection to pass to underlying scanners: user cols minus
/// system/`_distance`, with PKs appended for dedup/staleness.
pub fn build_scanner_projection(
    user_projection: Option<&[String]>,
    base_schema: &SchemaRef,
    pk_columns: &[String],
) -> Vec<String> {
    let mut cols: Vec<String> = if let Some(p) = user_projection {
        p.iter().filter(|c| !is_auto_managed(c)).cloned().collect()
    } else {
        base_schema
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect()
    };

    for pk in pk_columns {
        if !cols.contains(pk) {
            cols.push(pk.clone());
        }
    }

    cols
}

/// Validate user-facing projection names before constructing a canonical schema.
pub fn validate_projection_names(
    user_projection: Option<&[String]>,
    base_schema: &SchemaRef,
    extra_columns: &[&str],
) -> Result<()> {
    let Some(user_projection) = user_projection else {
        return Ok(());
    };
    let lance_schema = LanceSchema::try_from(base_schema.as_ref())?;
    for name in user_projection {
        if is_system_column(name) || extra_columns.contains(&name.as_str()) {
            continue;
        }
        // `resolve` walks the dotted path and reports whether every segment
        // exists. `project` is not a substitute: it errors on a missing
        // top-level column but yields an empty struct for a missing *child*,
        // so `meta.nope` would slip through.
        if lance_schema.resolve(name.as_str()).is_none() {
            return Err(lance_core::Error::invalid_input(format!(
                "Column '{}' not found in schema",
                name
            )));
        }
    }
    Ok(())
}

/// Canonical output schema honoring user column order.
///
/// System cols → nullable `UInt64` at user position (filled by
/// `project_to_canonical`). `_distance` (when `include_distance`) →
/// nullable `Float32` at user position, appended if absent. PKs appended.
///
/// Nested paths resolve to their narrowed struct: `meta.a` contributes
/// `meta: Struct<a>`, and sibling leaves collapse into one field at the
/// parent's first-mentioned position.
pub fn canonical_output_schema(
    user_projection: Option<&[String]>,
    base_schema: &SchemaRef,
    pk_columns: &[String],
    include_distance: bool,
) -> Result<SchemaRef> {
    let mut ordered: Vec<String> = if let Some(p) = user_projection {
        p.to_vec()
    } else {
        base_schema
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect()
    };

    for pk in pk_columns {
        if !ordered.contains(pk) {
            ordered.push(pk.clone());
        }
    }

    if include_distance && !ordered.iter().any(|c| c == DISTANCE_COLUMN) {
        ordered.push(DISTANCE_COLUMN.to_string());
    }

    let data_names: Vec<String> = ordered
        .iter()
        .filter(|n| !is_auto_managed(n))
        .cloned()
        .collect();
    let mut by_name: HashMap<String, Arc<Field>> = resolve_data_fields(&data_names, base_schema)?
        .into_iter()
        .map(|f| (f.name().clone(), f))
        .collect();

    let mut fields: Vec<Arc<Field>> = Vec::with_capacity(ordered.len());
    for name in &ordered {
        if name == DISTANCE_COLUMN {
            if include_distance {
                fields.push(Arc::new(Field::new(
                    DISTANCE_COLUMN,
                    DataType::Float32,
                    true,
                )));
            }
        } else if is_system_column(name) {
            fields.push(Arc::new(Field::new(name.clone(), DataType::UInt64, true)));
        } else if let Some(field) = by_name.remove(&top_level_of(name)?) {
            // `remove` is what collapses a second mention of the same parent
            // (`meta.a` then `meta.c`) into the single merged field.
            fields.push(field);
        }
    }

    Ok(Arc::new(Schema::new(fields)))
}

/// Wrap `plan` so the named columns become typed NULL literals; all
/// other columns are forwarded unchanged. Schema is preserved (same
/// fields, same dtypes). Useful for stripping the *value* of an
/// internal column after it has served its purpose (e.g. `_rowaddr`
/// after the per-arm local sort) without breaking downstream schema
/// matching.
pub fn null_columns(
    plan: Arc<dyn ExecutionPlan>,
    names: &[&str],
) -> Result<Arc<dyn ExecutionPlan>> {
    let input_schema = plan.schema();
    let mut project_exprs: Vec<(Arc<dyn PhysicalExpr>, String)> =
        Vec::with_capacity(input_schema.fields().len());
    for (idx, field) in input_schema.fields().iter().enumerate() {
        let name = field.name();
        let expr: Arc<dyn PhysicalExpr> = if names.contains(&name.as_str()) {
            Arc::new(Literal::new(
                ScalarValue::try_from(field.data_type()).map_err(|e| {
                    lance_core::Error::internal(format!(
                        "Cannot build NULL literal for {}: {}",
                        field.data_type(),
                        e
                    ))
                })?,
            ))
        } else {
            Arc::new(Column::new(name, idx))
        };
        project_exprs.push((expr, name.clone()));
    }
    let projection_exec = ProjectionExec::try_new(project_exprs, plan).map_err(|e| {
        lance_core::Error::internal(format!(
            "Failed to build null_columns ProjectionExec: {}",
            e
        ))
    })?;
    Ok(Arc::new(projection_exec))
}

/// Force `plan` to report exactly `target_schema`; a no-op when they agree.
///
/// `ProjectionExec` derives its nullability from the expressions, so the
/// storage schema's widened columns leave the WAL arms disagreeing with the
/// base arm — which `CoalesceFirstExec` and `concat_batches` both reject.
pub(super) fn force_schema(
    plan: Arc<dyn ExecutionPlan>,
    target_schema: &SchemaRef,
) -> Arc<dyn ExecutionPlan> {
    if plan.schema() == *target_schema {
        return plan;
    }
    Arc::new(SchemaRelabelExec::new(plan, target_schema.clone()))
}

/// Wrap `plan` to emit exactly `target_schema`. Source columns are
/// forwarded by name; system / `_distance` cols missing from the source
/// are NULL-filled. Other missing columns are an internal error.
///
/// Reports `target_schema` exactly, nullability included — see [`force_schema`].
pub fn project_to_canonical(
    plan: Arc<dyn ExecutionPlan>,
    target_schema: &SchemaRef,
) -> Result<Arc<dyn ExecutionPlan>> {
    let input_schema = plan.schema();
    let mut project_exprs: Vec<(Arc<dyn PhysicalExpr>, String)> =
        Vec::with_capacity(target_schema.fields().len());
    for field in target_schema.fields() {
        let name = field.name();
        let expr: Arc<dyn PhysicalExpr> = match input_schema.column_with_name(name) {
            Some((idx, _)) => Arc::new(Column::new(name, idx)),
            None if is_system_column(name) => Arc::new(Literal::new(ScalarValue::UInt64(None))),
            None if name == DISTANCE_COLUMN => Arc::new(Literal::new(ScalarValue::Float32(None))),
            None => {
                return Err(lance_core::Error::internal(format!(
                    "Column '{}' missing from canonical projection source schema (have: {:?})",
                    name,
                    input_schema
                        .fields()
                        .iter()
                        .map(|f| f.name().clone())
                        .collect::<Vec<_>>()
                )));
            }
        };
        project_exprs.push((expr, name.clone()));
    }
    let projection_exec = ProjectionExec::try_new(project_exprs, plan).map_err(|e| {
        lance_core::Error::internal(format!("Failed to build canonical ProjectionExec: {}", e))
    })?;
    Ok(force_schema(Arc::new(projection_exec), target_schema))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::RecordBatch;
    use arrow_schema::Schema as ArrowSchema;
    use datafusion_physical_plan::test::TestMemoryExec;

    fn schema() -> SchemaRef {
        Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("vector", DataType::Float32, true),
        ]))
    }

    /// [`schema`] as `relax_non_pk_nullability` would leave it: `id` widened.
    fn widened_schema() -> SchemaRef {
        Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, true),
            Field::new("name", DataType::Utf8, true),
            Field::new("vector", DataType::Float32, true),
        ]))
    }

    fn plan_emitting(schema: SchemaRef) -> Arc<dyn ExecutionPlan> {
        let batch = RecordBatch::new_empty(schema.clone());
        TestMemoryExec::try_new_exec(&[vec![batch]], schema, None).unwrap()
    }

    #[test]
    fn scanner_projection_strips_system_and_distance() {
        let s = schema();
        let pks = vec!["id".to_string()];
        let user = vec![
            "_distance".to_string(),
            "vector".to_string(),
            "_rowid".to_string(),
            "_rowaddr".to_string(),
        ];
        let cols = build_scanner_projection(Some(&user), &s, &pks);
        assert_eq!(cols, vec!["vector".to_string(), "id".to_string()]);
    }

    #[test]
    fn scanner_projection_default_uses_base_schema() {
        let s = schema();
        let pks = vec!["id".to_string()];
        let cols = build_scanner_projection(None, &s, &pks);
        assert_eq!(
            cols,
            vec!["id".to_string(), "name".to_string(), "vector".to_string()]
        );
    }

    #[test]
    fn canonical_schema_honors_user_order_for_distance() {
        let s = schema();
        let pks = vec!["id".to_string()];
        let user = vec!["_distance".to_string(), "vector".to_string()];
        let out = canonical_output_schema(Some(&user), &s, &pks, true).unwrap();
        let names: Vec<&str> = out.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names, vec!["_distance", "vector", "id"]);
        assert_eq!(
            out.field_with_name("_distance").unwrap().data_type(),
            &DataType::Float32
        );
    }

    #[test]
    fn canonical_schema_includes_system_cols_as_nullable_uint64() {
        let s = schema();
        let pks = vec!["id".to_string()];
        let user = vec![
            "vector".to_string(),
            "_rowid".to_string(),
            "_rowaddr".to_string(),
            "_rowoffset".to_string(),
        ];
        let out = canonical_output_schema(Some(&user), &s, &pks, false).unwrap();
        let names: Vec<&str> = out.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(
            names,
            vec!["vector", "_rowid", "_rowaddr", "_rowoffset", "id"]
        );
        for sys in ["_rowid", "_rowaddr", "_rowoffset"] {
            let field = out.field_with_name(sys).unwrap();
            assert_eq!(field.data_type(), &DataType::UInt64);
            assert!(field.is_nullable(), "{sys} must be nullable for NULL fill");
        }
    }

    #[test]
    fn canonical_schema_appends_distance_when_missing() {
        let s = schema();
        let pks = vec!["id".to_string()];
        let user = vec!["vector".to_string()];
        let out = canonical_output_schema(Some(&user), &s, &pks, true).unwrap();
        let names: Vec<&str> = out.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names, vec!["vector", "id", "_distance"]);
    }

    #[test]
    fn canonical_schema_drops_distance_when_not_requested() {
        let s = schema();
        let pks = vec!["id".to_string()];
        let user = vec!["_distance".to_string(), "vector".to_string()];
        let out = canonical_output_schema(Some(&user), &s, &pks, false).unwrap();
        let names: Vec<&str> = out.fields().iter().map(|f| f.name().as_str()).collect();
        // _distance dropped because include_distance=false (e.g. point lookup / scan).
        assert_eq!(names, vec!["vector", "id"]);
    }

    /// `meta: Struct<a, b, c>` alongside flat columns.
    fn nested_schema() -> SchemaRef {
        use arrow_schema::Fields;
        Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "meta",
                DataType::Struct(Fields::from(vec![
                    Field::new("a", DataType::Int64, true),
                    Field::new("b", DataType::Utf8, true),
                    Field::new("c", DataType::Float64, true),
                ])),
                true,
            ),
        ]))
    }

    fn struct_children(schema: &SchemaRef, name: &str) -> Vec<String> {
        match schema.field_with_name(name).unwrap().data_type() {
            DataType::Struct(fields) => fields.iter().map(|f| f.name().clone()).collect(),
            other => panic!("{name} is not a struct: {other:?}"),
        }
    }

    #[test]
    fn canonical_schema_narrows_a_struct_to_the_selected_leaf() {
        let s = nested_schema();
        let pks = vec!["id".to_string()];
        let user = vec!["meta.a".to_string()];
        let out = canonical_output_schema(Some(&user), &s, &pks, false).unwrap();
        let names: Vec<&str> = out.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names, vec!["meta", "id"]);
        assert_eq!(struct_children(&out, "meta"), vec!["a".to_string()]);
    }

    #[test]
    fn canonical_schema_merges_sibling_leaves_at_the_parents_position() {
        let s = nested_schema();
        let pks = vec!["id".to_string()];
        // Two leaves of one parent must collapse into a single `meta` column,
        // seated where the parent was first mentioned.
        let user = vec!["meta.a".to_string(), "id".to_string(), "meta.c".to_string()];
        let out = canonical_output_schema(Some(&user), &s, &pks, false).unwrap();
        let names: Vec<&str> = out.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names, vec!["meta", "id"]);
        assert_eq!(
            struct_children(&out, "meta"),
            vec!["a".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn canonical_schema_keeps_a_whole_struct_whole() {
        let s = nested_schema();
        let pks = vec!["id".to_string()];
        let user = vec!["meta".to_string()];
        let out = canonical_output_schema(Some(&user), &s, &pks, false).unwrap();
        assert_eq!(
            struct_children(&out, "meta"),
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn validate_accepts_a_nested_path_and_rejects_a_bogus_child() {
        let s = nested_schema();
        validate_projection_names(Some(&["meta.a".to_string()]), &s, &[]).unwrap();
        let err = validate_projection_names(Some(&["meta.nope".to_string()]), &s, &[]).unwrap_err();
        assert!(
            err.to_string().contains("meta.nope"),
            "error should name the bad column, got: {err}"
        );
    }

    #[test]
    fn force_schema_leaves_a_matching_plan_alone() {
        let plan = plan_emitting(schema());
        let forced = force_schema(plan.clone(), &schema());
        assert!(
            Arc::ptr_eq(&plan, &forced),
            "a plan already reporting the target schema must not be wrapped"
        );
    }

    #[test]
    fn force_schema_relabels_a_nullability_mismatch() {
        let forced = force_schema(plan_emitting(widened_schema()), &schema());
        assert_eq!(forced.name(), "SchemaRelabelExec");
        assert_eq!(forced.schema(), schema());
    }

    #[test]
    fn project_to_canonical_reports_the_target_schema() {
        // The ProjectionExec alone would follow its input and report `id` as
        // nullable; the relabel is what pins the output to the target.
        let target = schema();
        let plan = project_to_canonical(plan_emitting(widened_schema()), &target).unwrap();
        assert_eq!(plan.schema(), target);
    }
}
