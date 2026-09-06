// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Read-path resolution of an index's covering ("included") columns.
//!
//! # Declaration versus capability
//!
//! Covering has two deliberately independent sources of truth:
//!
//! * the **manifest declaration** — `IndexMetadata::covering_fields`, a dependency list of
//!   dataset field ids in declaration order;
//! * the **physical capability** — columns the selected segment's storage schema proves it
//!   can serve, including their source dataset field ids.
//!
//! The declaration is not evidence that the payload exists. Transitional writers and
//! lifecycle operations may preserve the dependency while omitting or withdrawing the
//! physical values. Readers may skip a base-table take only for the declaration/capability
//! intersection proven across every selected segment.
//!
//! This module resolves only the manifest side. Storage-aware planners must narrow its
//! result through [`VectorIndex::physical_covering_fields`] before adding columns to an
//! executable output schema.
//!
//! Nothing here is vector-specific: `covering_fields` lives on `IndexMetadata`, so the
//! FTS execs resolve covering exactly the same way and must not re-derive it.
//!
//! [`VectorIndex::physical_covering_fields`]: lance_index::vector::VectorIndex::physical_covering_fields

use arrow_schema::Field as ArrowField;
use lance_core::datatypes::Schema;
use lance_core::{Error, Result};

/// The logically requested covering columns for one query: everything the manifest
/// declares, narrowed to that query's covering projection.
///
/// `projection` is [`Query::covering_projection`], whose three states this function is
/// the primary consumer of:
///
/// | `projection`  | result                                       |
/// |---------------|----------------------------------------------|
/// | `None`        | every declared covering column               |
/// | `Some(&[])`   | empty — the caller must do no covering work  |
/// | `Some(cols)`  | the declared columns named in `cols`         |
///
/// An empty result is a *contract*, not an incidental outcome: it means the caller skips
/// the covering read altogether rather than projecting an already-loaded batch down to
/// zero columns. Note that `None` and `Some(&[])` therefore differ enormously in cost
/// while being indistinguishable in results.
///
/// Returns resolved Arrow **fields**, not ids. This is not yet an executable capability:
/// callers that plan a storage read must intersect these fields with the physical segment
/// schemas before declaring them in an output plan.
///
/// # Order
///
/// The result follows `covering_fields`, never the order of `projection`. That is the
/// logical output order. Physical storage must prove that it can emit the selected fields
/// in a compatible order; otherwise the reader falls back to the base table.
///
/// # Errors
///
/// A declared id absent from the dataset schema's top-level fields is an error, and is
/// reported even when `projection` would have narrowed that column away. The manifest is
/// authoritative for dependencies: such a mismatch means index metadata and schema
/// disagree, and letting a narrowing decision suppress it would turn a corrupt index into
/// one that appears to work for some queries. For the same reason callers must run any cross-segment
/// agreement checks on the *declared* sets before calling this — narrowing first can
/// reduce genuinely disagreeing segments to a subset on which they happen to agree.
///
/// [`Query::covering_projection`]: lance_index::vector::Query::covering_projection
pub fn effective_covering(
    covering_fields: &[i32],
    projection: Option<&[String]>,
    schema: &Schema,
) -> Result<Vec<ArrowField>> {
    Ok(
        effective_covering_with_ids(covering_fields, projection, schema)?
            .into_iter()
            .map(|(_, field)| field)
            .collect(),
    )
}

/// [`effective_covering`] with each resolved Arrow field still paired to the manifest
/// field id that selected it. Storage-capability checks use this form so logical identity
/// is never reconstructed from a column name.
pub fn effective_covering_with_ids(
    covering_fields: &[i32],
    projection: Option<&[String]>,
    schema: &Schema,
) -> Result<Vec<(i32, ArrowField)>> {
    if covering_fields.is_empty() {
        return Ok(Vec::new());
    }
    let mut fields = Vec::with_capacity(covering_fields.len());
    for id in covering_fields {
        // Top-level lookup only, deliberately not `Schema::field_by_id`: that recurses
        // into struct children, and a nested field matched here would be emitted under
        // its leaf name -- a name a covered projection cannot address, and one that can
        // collide with an unrelated top-level column. Covering columns are top-level by
        // construction; `validate_covering_columns` rejects dotted paths at create time.
        let field = schema
            .fields
            .iter()
            .find(|field| field.id == *id)
            .ok_or_else(|| {
                Error::index(format!(
                    "index declares covering field id {id}, which is not present as a \
                     top-level field in the current dataset schema; index metadata and \
                     schema are inconsistent"
                ))
            })?;
        // Convert just this field. `ArrowSchema::from(schema)` is exactly this conversion
        // mapped over every field, but it walks the whole table (including nested
        // structs) on every covered plan.
        if projection.is_none_or(|wanted| wanted.iter().any(|name| name == &field.name)) {
            fields.push((*id, ArrowField::from(field)));
        }
    }
    Ok(fields)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::{DataType, Field as ArrowField, Fields, Schema as ArrowSchema};

    /// `price` and `payload` are the covering candidates; `nested.price` exists to prove
    /// that a nested field sharing a top-level name is never what gets resolved.
    fn schema() -> Schema {
        let arrow = ArrowSchema::new(vec![
            ArrowField::new("vec", DataType::Int32, false),
            ArrowField::new("price", DataType::Int32, true),
            ArrowField::new("payload", DataType::Utf8, true),
            ArrowField::new(
                "nested",
                DataType::Struct(Fields::from(vec![ArrowField::new(
                    "price",
                    DataType::Float64,
                    true,
                )])),
                true,
            ),
        ]);
        Schema::try_from(&arrow).unwrap()
    }

    fn names(fields: &[ArrowField]) -> Vec<&str> {
        fields.iter().map(|f| f.name().as_str()).collect()
    }

    fn field_id(schema: &Schema, name: &str) -> i32 {
        schema.field(name).unwrap().id
    }

    #[test]
    fn no_projection_keeps_every_declared_column() {
        let s = schema();
        let ids = vec![field_id(&s, "price"), field_id(&s, "payload")];
        let got = effective_covering(&ids, None, &s).unwrap();
        assert_eq!(
            names(&got),
            vec!["price", "payload"],
            "`None` means no narrowing was computed, so every declared column is emitted"
        );
    }

    #[test]
    fn projection_narrows_to_what_the_query_needs() {
        let s = schema();
        let ids = vec![field_id(&s, "price"), field_id(&s, "payload")];
        let projection = vec!["payload".to_string()];
        let got = effective_covering(&ids, Some(&projection), &s).unwrap();
        assert_eq!(names(&got), vec!["payload"]);
    }

    /// The state the whole feature exists for. It is distinct from `None`, which yields
    /// the full declared set -- conflating them silently restores full materialization.
    #[test]
    fn empty_projection_is_not_the_same_as_no_projection() {
        let s = schema();
        let ids = vec![field_id(&s, "price"), field_id(&s, "payload")];
        let narrowed = effective_covering(&ids, Some(&[]), &s).unwrap();
        assert!(
            narrowed.is_empty(),
            "`Some(&[])` means this query needs no covering column at all"
        );
        let unnarrowed = effective_covering(&ids, None, &s).unwrap();
        assert_eq!(
            names(&unnarrowed),
            vec!["price", "payload"],
            "`None` must NOT behave like `Some(&[])`; if these two agree the seam is dead"
        );
    }

    /// Emission order is the index's declaration order, never the caller's request order
    /// and never schema order. Both alternatives pair values with the wrong names.
    ///
    /// The declared order below is deliberately the reverse of the schema's ascending
    /// field-id order (vec=0, price=1, payload=2, ...), and the requested order is
    /// deliberately different again. An implementation that walked `schema.fields()` and
    /// filtered by membership, or that echoed the request order, produces `["price",
    /// "payload"]` and fails. Do not "tidy" either list back into agreement.
    #[test]
    fn preserves_declaration_order_not_schema_or_request_order() {
        let s = schema();
        let declared = vec![field_id(&s, "payload"), field_id(&s, "price")];
        let requested = vec!["price".to_string(), "payload".to_string()];
        let got = effective_covering(&declared, Some(&requested), &s).unwrap();
        assert_eq!(
            names(&got),
            vec!["payload", "price"],
            "declaration order is storage order; any other order mismatches the batch \
             the storage produces"
        );

        // Permuting only the declaration must permute only the result: that is what
        // makes this test sensitive to order rather than to membership.
        let permuted = vec![field_id(&s, "price"), field_id(&s, "payload")];
        let got = effective_covering(&permuted, Some(&requested), &s).unwrap();
        assert_eq!(names(&got), vec!["price", "payload"]);
    }

    #[test]
    fn declared_columns_absent_from_the_projection_are_dropped_entirely() {
        let s = schema();
        let ids = vec![field_id(&s, "price")];
        let projection = vec!["vec".to_string()];
        assert!(
            effective_covering(&ids, Some(&projection), &s)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn a_plain_index_declares_nothing_in_every_projection_state() {
        let s = schema();
        for projection in [None, Some(&[][..]), Some(&["price".to_string()][..])] {
            assert!(effective_covering(&[], projection, &s).unwrap().is_empty());
        }
    }

    /// A nested field can carry the same leaf name as a top-level column. Resolving it
    /// would emit a `Float64` column named `price` where the covered projection expects
    /// the top-level `Int32` one.
    #[test]
    fn resolves_top_level_fields_only() {
        let s = schema();
        let nested_price_id = s
            .field("nested")
            .unwrap()
            .children
            .iter()
            .find(|f| f.name == "price")
            .unwrap()
            .id;
        let err = effective_covering(&[nested_price_id], None, &s)
            .expect_err("a nested field id is not a valid covering declaration");
        assert!(matches!(err, Error::Index { .. }));

        let top_level = effective_covering(&[field_id(&s, "price")], None, &s).unwrap();
        assert_eq!(top_level[0].data_type(), &DataType::Int32);
    }

    /// Metadata that disagrees with the schema must fail loudly even when the narrowing
    /// would have discarded the offending column: a corrupt index must not appear to work
    /// for the subset of queries that happen not to ask for the broken column.
    #[test]
    fn unresolvable_declared_id_errors_even_when_narrowed_away() {
        let s = schema();
        let ids = vec![9999, field_id(&s, "price")];
        for projection in [None, Some(&[][..]), Some(&["price".to_string()][..])] {
            let err = effective_covering(&ids, projection, &s)
                .expect_err("a declared covering id absent from the schema must be rejected");
            assert!(matches!(err, Error::Index { .. }));
            assert!(
                err.to_string().contains("9999"),
                "error names the offending id, got: {err}"
            );
        }
    }
}
