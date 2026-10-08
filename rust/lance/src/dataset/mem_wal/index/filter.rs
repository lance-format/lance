// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! How a filter expression reaches a memtable index.
//!
//! The memtable does not define a predicate vocabulary of its own. It hands the
//! filter to [`apply_scalar_indices`], the same pass the base table's scan uses,
//! with a provider built from the memtable's own indexes. What comes back is a
//! tree of index searches and a leftover expression, and the tree's leaves are
//! `AnyQuery` values the indexes answer directly.
//!
//! So the memtable's filter capability comes from the plugins' parsers rather
//! than from code here:
//!
//! * Every expression shape a plugin's [`ScalarQueryParser`] claims — ranges
//!   with either bound inclusive, `IN`, `IS NULL`, `LIKE 'prefix%'`, a spatial
//!   or text function — reaches the memtable index, because it is literally the
//!   parser the on-disk index uses.
//! * `AND` and `OR` across different columns and different indexes compose.
//! * What no index can answer comes back as the leftover expression and is
//!   applied as an ordinary filter over the rows the indexes narrowed to.
//!
//! `NOT` is deliberately not evaluated here: complementing a result needs the
//! rows whose value is null tracked separately, to keep SQL's three-valued
//! logic, and the memtable's indexes do not report that yet. A filter with a
//! `NOT` over an indexed leaf falls back to a scan rather than answering it
//! wrongly.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_schema::DataType;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::logical_expr::{Cast, Expr};
use lance_core::Result;
use lance_core::datatypes::Schema as LanceSchema;
use lance_index::scalar::expression::{
    IndexInformationProvider, IndexedExpression, MultiQueryParser, ScalarIndexExpr,
    apply_scalar_indices,
};

use super::query::{MemMatches, MemSearchResult, PositionSet, ScalarQuery, SearchContext};
use super::{IndexStore, MemIndexSpec};

/// The memtable's indexes, in the shape the expression pass expects.
///
/// Built once when a memtable is configured: the parsers come from the plugins,
/// so a deployment's own index kind claims expressions here without Lance
/// knowing it exists.
#[derive(Debug, Default)]
pub struct MemIndexCatalog {
    columns: HashMap<String, (DataType, MultiQueryParser)>,
}

impl MemIndexCatalog {
    /// Collect the parsers for every spec that has one.
    ///
    /// A spec whose plugin returns no parser is simply absent here, which is
    /// how a vector or full-text index stays out of filter planning.
    pub fn new(specs: &[MemIndexSpec], schema: &LanceSchema) -> Self {
        let mut columns: HashMap<String, (DataType, MultiQueryParser)> = HashMap::new();
        for spec in specs {
            let details = spec.details.as_deref();
            // A multi-column index claims expressions on each column it
            // covers, and each column needs its own parser.
            for column in &spec.columns {
                let Some(field) = schema.field(column) else {
                    continue;
                };
                let Some(parser) = spec.plugin.query_parser(spec.name.clone(), details) else {
                    continue;
                };
                match columns.get_mut(column) {
                    // Two indexes on one column: the first to claim an
                    // expression answers it, which is what the base table does.
                    Some((_, existing)) => existing.add(parser),
                    None => {
                        columns.insert(
                            column.clone(),
                            (field.data_type(), MultiQueryParser::single(parser)),
                        );
                    }
                }
            }
        }
        Self { columns }
    }

    /// The provider for a memtable maintaining `specs`.
    pub fn for_specs(specs: &[MemIndexSpec], schema: &LanceSchema) -> Arc<Self> {
        Arc::new(Self::new(specs, schema))
    }

    /// Whether any index claims expressions on some column.
    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }
}

impl IndexInformationProvider for MemIndexCatalog {
    fn get_index(&self, col: &str) -> Option<(&DataType, &MultiQueryParser)> {
        self.columns
            .get(col)
            .map(|(data_type, parser)| (data_type, parser))
    }
}

/// Split `filter` into index searches and whatever is left to evaluate.
///
/// `None` when no index can help, which is the caller's signal to plan an
/// ordinary scan.
pub fn plan_filter(filter: &Expr, catalog: &MemIndexCatalog) -> Result<Option<IndexedExpression>> {
    if catalog.is_empty() {
        return Ok(None);
    }
    let split = apply_scalar_indices(see_through_relabelling(filter, catalog), catalog)?;
    let Some(query) = &split.scalar_query else {
        return Ok(None);
    };
    // A tree the memtable cannot evaluate is worse than no tree: it would have
    // to scan anyway, and planning an index node first only adds a step.
    if !is_evaluable(query) {
        return Ok(None);
    }
    Ok(Some(split))
}

/// `filter` with each cast of an indexed column removed when the cast changes
/// only the column's nested field names or metadata.
///
/// A memtable's schema carries Lance field ids on nested fields, so comparing a
/// list column with a list literal coerces the column to the literal's type —
/// a cast that leaves every value as it was, but hides the column from the
/// parser that would otherwise claim the expression. The filter evaluated over
/// the rows keeps its cast; only the index split looks through it.
fn see_through_relabelling(filter: &Expr, catalog: &MemIndexCatalog) -> Expr {
    filter
        .clone()
        .transform_up(|expr| {
            if let Expr::Cast(Cast { expr: inner, field }) = &expr
                && let Expr::Column(column) = inner.as_ref()
                && catalog
                    .get_index(&column.name)
                    .is_some_and(|(column_type, _)| column_type.equals_datatype(field.data_type()))
            {
                return Ok(Transformed::yes(inner.as_ref().clone()));
            }
            Ok(Transformed::no(expr))
        })
        .map_or_else(|_| filter.clone(), |transformed| transformed.data)
}

/// Whether the memtable can evaluate this tree. See the module note on `NOT`.
fn is_evaluable(expr: &ScalarIndexExpr) -> bool {
    match expr {
        ScalarIndexExpr::Not(_) => false,
        ScalarIndexExpr::And(lhs, rhs) | ScalarIndexExpr::Or(lhs, rhs) => {
            is_evaluable(lhs) && is_evaluable(rhs)
        }
        ScalarIndexExpr::Query(_) => true,
    }
}

/// Evaluate a tree of index searches against one memtable.
///
/// The set algebra is elementwise on the two endpoints, so an exact leaf
/// combined with a narrowing one yields a narrowing result, and the caller
/// re-checks exactly when it has to.
pub fn evaluate(
    expr: &ScalarIndexExpr,
    indexes: &IndexStore,
    ctx: &SearchContext,
) -> Result<MemSearchResult> {
    match expr {
        ScalarIndexExpr::And(lhs, rhs) => {
            Ok(evaluate(lhs, indexes, ctx)? & evaluate(rhs, indexes, ctx)?)
        }
        ScalarIndexExpr::Or(lhs, rhs) => {
            Ok(evaluate(lhs, indexes, ctx)? | evaluate(rhs, indexes, ctx)?)
        }
        // Rejected while planning; a tree reaching here has none.
        ScalarIndexExpr::Not(_) => Ok(unknown(ctx)),
        ScalarIndexExpr::Query(search) => {
            let Some(index) = indexes.get_index(&search.index_name) else {
                return Ok(unknown(ctx));
            };
            match index.search(&ScalarQuery(search.query.as_ref()), ctx)? {
                Some(MemMatches::Filter(result)) => {
                    let result = result.truncate_to(ctx.max_visible);
                    // The index answered exactly, but the expression pass asked
                    // for a re-check anyway — a parser says so when the query it
                    // built is a widening of the expression it came from.
                    Ok(if search.needs_recheck {
                        MemSearchResult::at_most(result.at_most)
                    } else {
                        result
                    })
                }
                // A ranked answer to a filter, or no answer at all: this index
                // cannot narrow the search, so nothing is ruled out.
                Some(MemMatches::Ranked(_)) | None => Ok(unknown(ctx)),
            }
        }
    }
}

/// Nothing is ruled out: every visible row is a candidate.
fn unknown(ctx: &SearchContext) -> MemSearchResult {
    MemSearchResult::at_most(PositionSet::all_visible(ctx.max_visible))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The positions a tree selected, and whether the caller must re-check them.
    fn positions(result: MemSearchResult) -> (Vec<u64>, bool) {
        let exact = result.is_exact();
        (result.at_most.into(), exact)
    }
    use arrow_array::{Int32Array, RecordBatch, StringArray};
    use arrow_schema::{Field, Schema as ArrowSchema};
    use datafusion::common::ScalarValue;
    use lance_datafusion::planner::Planner;
    use lance_index::scalar::SargableQuery;

    use crate::dataset::mem_wal::index::{IndexStore, MemIndexSpec};

    /// A list column whose nested field carries a field id is cast to a list
    /// literal's type when compared with one. The cast changes no value, so a
    /// label-list parser must still claim the comparison.
    #[test]
    fn an_index_sees_through_a_cast_that_only_relabels_nested_fields() {
        use lance_index::scalar::expression::LabelListQueryParser;
        use std::collections::HashMap as Map;

        let item = Field::new("item", arrow_schema::DataType::Utf8, true)
            .with_metadata(Map::from([("lance:field_id".to_string(), "3".to_string())]));
        let tags = arrow_schema::DataType::List(Arc::new(item));
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "tags",
            tags.clone(),
            true,
        )]));
        let planner = Planner::new(schema);
        let filter = planner
            .optimize_expr(
                planner
                    .parse_filter("array_has_any(tags, make_array('t7'))")
                    .unwrap(),
            )
            .unwrap();
        assert!(
            filter.to_string().contains("CAST(tags"),
            "the comparison must cast the column for this test to mean anything: {filter}"
        );

        let catalog = MemIndexCatalog {
            columns: HashMap::from([(
                "tags".to_string(),
                (
                    tags,
                    MultiQueryParser::single(Box::new(LabelListQueryParser::new(
                        "tags_idx".to_string(),
                        "LabelList".to_string(),
                    ))),
                ),
            )]),
        };
        let split = plan_filter(&filter, &catalog).unwrap();
        assert!(split.is_some(), "the label list must claim {filter}");
    }

    fn schema() -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![
            Field::new("id", arrow_schema::DataType::Int32, false),
            Field::new("name", arrow_schema::DataType::Utf8, true),
            Field::new("other", arrow_schema::DataType::Int32, true),
        ]))
    }

    /// Two B-trees, on `id` and `name`, over ten rows: `id` counts up and
    /// `name` is `alpha<id>`. No underscore in the names, because `_` is a
    /// LIKE wildcard and would make the prefix case test something else.
    fn store() -> (IndexStore, Vec<MemIndexSpec>) {
        let arrow = schema();
        let lance = LanceSchema::try_from(arrow.as_ref()).unwrap();
        let specs = vec![
            MemIndexSpec::btree("id_idx", 0, "id"),
            MemIndexSpec::btree("name_idx", 1, "name"),
        ];
        let store = IndexStore::from_specs(&specs, &lance, 1_000, 16).unwrap();

        let ids: Vec<i32> = (0..10).collect();
        let names: Vec<String> = ids.iter().map(|id| format!("alpha{id}")).collect();
        let others: Vec<i32> = ids.iter().map(|id| id % 3).collect();
        let batch = RecordBatch::try_new(
            arrow,
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(StringArray::from(names)),
                Arc::new(Int32Array::from(others)),
            ],
        )
        .unwrap();
        store.insert(&batch, 0).unwrap();
        (store, specs)
    }

    fn plan(filter: &str, specs: &[MemIndexSpec]) -> Option<IndexedExpression> {
        let arrow = schema();
        let lance = LanceSchema::try_from(arrow.as_ref()).unwrap();
        let catalog = MemIndexCatalog::new(specs, &lance);
        let planner = Planner::new(arrow);
        let expr = planner
            .optimize_expr(planner.parse_filter(filter).unwrap())
            .unwrap();
        plan_filter(&expr, &catalog).unwrap()
    }

    fn run(filter: &str) -> (Vec<u64>, bool) {
        let (store, specs) = store();
        let split = plan(filter, &specs).expect("the filter reaches an index");
        let query = split.scalar_query.expect("an index search");
        positions(evaluate(&query, &store, &SearchContext::new(u64::MAX)).unwrap())
    }

    /// Every shape the on-disk B-tree's parser claims reaches the memtable one.
    #[test]
    fn a_btree_answers_every_shape_its_parser_claims() {
        for (filter, expected) in [
            ("id = 5", vec![5]),
            ("id IN (2, 5, 8)", vec![2, 5, 8]),
            ("id < 3", vec![0, 1, 2]),
            ("id <= 2", vec![0, 1, 2]),
            ("id > 7", vec![8, 9]),
            ("id >= 8", vec![8, 9]),
            ("id BETWEEN 4 AND 6", vec![4, 5, 6]),
            // A prefix match, turned into a range over the ordered keys.
            ("name LIKE 'alpha1%'", vec![1]),
        ] {
            let (positions, exact) = run(filter);
            assert_eq!(positions, expected, "filter: {filter}");
            assert!(exact, "a B-tree decides, it does not narrow: {filter}");
        }
    }

    /// `AND` and `OR` across two different indexes compose. The memtable used
    /// to take one predicate on one column and scan for anything else.
    #[test]
    fn compound_filters_combine_two_indexes() {
        let (positions, exact) = run("id >= 4 AND name = 'alpha5'");
        assert_eq!(positions, vec![5]);
        assert!(exact);

        let (positions, exact) = run("id = 1 OR name = 'alpha7'");
        assert_eq!(positions, vec![1, 7]);
        assert!(exact);
    }

    /// A conjunct no index covers is not a reason to give up the ones that are
    /// covered: the indexed half narrows, and the rest comes back as the
    /// expression to apply afterwards.
    #[test]
    fn an_unindexed_conjunct_becomes_the_leftover_expression() {
        let (_, specs) = store();
        let split =
            plan("id >= 4 AND other = 1", &specs).expect("the indexed half reaches an index");
        assert!(
            split.scalar_query.is_some(),
            "the `id` half is answered by its index"
        );
        assert!(
            split.refine_expr.is_some(),
            "the `other` half has no index and must be left to the filter"
        );
    }

    /// Nothing is indexed, so there is nothing to plan and the caller scans.
    #[test]
    fn a_filter_on_no_indexed_column_declines() {
        let (_, specs) = store();
        assert!(plan("other = 1", &specs).is_none());
    }

    /// Complementing a result needs the rows whose value is null tracked
    /// separately, to keep SQL's three-valued logic. Until an index reports
    /// that, answering `NOT` from the index would drop or admit null rows, so
    /// the whole filter falls back to a scan.
    #[test]
    fn a_negated_filter_declines_rather_than_answering_wrongly() {
        let (_, specs) = store();
        assert!(plan("NOT (id = 5)", &specs).is_none());
    }

    /// Planning runs on the optimized expression, so a float zero reaches the
    /// index as the two-element set IEEE 754 says it is rather than as one
    /// bit-exact key — the same answer the full scan beside it gives.
    #[test]
    fn planning_sees_the_optimized_expression() {
        let arrow = Arc::new(ArrowSchema::new(vec![Field::new(
            "value",
            arrow_schema::DataType::Float64,
            true,
        )]));
        let lance = LanceSchema::try_from(arrow.as_ref()).unwrap();
        let specs = vec![MemIndexSpec::btree("value_idx", 0, "value")];
        let catalog = MemIndexCatalog::new(&specs, &lance);
        let planner = Planner::new(arrow);

        for spelling in ["value = 0.0", "value = 0"] {
            let expr = planner
                .optimize_expr(planner.parse_filter(spelling).unwrap())
                .unwrap();
            let split = plan_filter(&expr, &catalog)
                .unwrap()
                .unwrap_or_else(|| panic!("{spelling} reaches the index"));
            let ScalarIndexExpr::Query(search) = split.scalar_query.unwrap() else {
                panic!("{spelling} should be one index search");
            };
            let query = search
                .query
                .as_any()
                .downcast_ref::<SargableQuery>()
                .expect("a sargable query");
            assert_eq!(
                query,
                &SargableQuery::IsIn(vec![
                    ScalarValue::Float64(Some(-0.0)),
                    ScalarValue::Float64(Some(0.0)),
                ]),
                "{spelling} must reach the index as both encodings of zero"
            );
        }
    }

    /// An index holds rows a reader must not see yet, because a writer runs
    /// ahead of the watermark that publishes them.
    #[test]
    fn evaluation_honors_the_visibility_watermark() {
        let (store, specs) = store();
        let split = plan("id >= 0", &specs).unwrap();
        let query = split.scalar_query.unwrap();
        let (positions, _) = positions(evaluate(&query, &store, &SearchContext::new(4)).unwrap());
        assert_eq!(positions, vec![0, 1, 2, 3, 4]);
    }

    /// An index the tree names but the store does not hold rules nothing out,
    /// rather than answering "no rows" and dropping every match.
    #[test]
    fn a_missing_index_rules_nothing_out() {
        let (store, _) = store();
        let missing = ScalarIndexExpr::Query(lance_index::scalar::expression::ScalarIndexSearch {
            column: "id".to_string(),
            index_name: "not_registered".to_string(),
            index_type: "BTree".to_string(),
            query: Arc::new(SargableQuery::Equals(ScalarValue::Int32(Some(5)))),
            needs_recheck: false,
            fragment_bitmap: None,
        });
        let (found, exact) = positions(evaluate(&missing, &store, &SearchContext::new(9)).unwrap());
        assert_eq!(found, (0..=9).collect::<Vec<_>>());
        assert!(!exact, "every row is a candidate the caller must re-check");
    }
}
