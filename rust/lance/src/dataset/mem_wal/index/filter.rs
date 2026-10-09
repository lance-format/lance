// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! How a filter expression reaches a memtable index.
//!
//! The filter goes through [`apply_scalar_indices`], the pass the base table's
//! scan uses, with a provider built from the memtable's indexes and their
//! plugins' [`ScalarQueryParser`]s. The result is a tree of index searches plus
//! a leftover expression applied to the rows the indexes narrowed to.
//!
//! `NOT` is not evaluated from indexes: complementing a result needs each
//! index's null rows, which they do not report, so such a filter is scanned.

use std::collections::HashMap;

use arrow_schema::DataType;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::logical_expr::{Cast, Expr};
use lance_core::Result;
use lance_core::datatypes::Schema as LanceSchema;
use lance_index::scalar::expression::{
    IndexInformationProvider, MultiQueryParser, ScalarIndexExpr, ScalarQueryParser,
    apply_scalar_indices,
};

use super::query::{MemMatches, MemSearchResult, PositionSet, ScalarQuery, SearchContext};
use super::{BTreeMemIndexPlugin, IndexStore, MemIndexPlugin, MemIndexSpec};

/// The name of the memtable's own key index: one no base-table index can have.
pub const OWN_KEY_INDEX: &str = "\u{0}primary key";

/// The memtable's indexes, in the shape the expression pass expects.
#[derive(Debug, Default)]
pub struct MemIndexCatalog {
    columns: HashMap<String, (DataType, MultiQueryParser)>,
    schema: Option<LanceSchema>,
}

impl MemIndexCatalog {
    /// The parsers of every single-column spec whose plugin has one. A parsed
    /// query names no column, so an index over several could not tell which
    /// one it is about.
    pub fn new(specs: &[MemIndexSpec], schema: &LanceSchema) -> Self {
        let mut catalog = Self {
            columns: HashMap::new(),
            schema: Some(schema.clone()),
        };
        for spec in specs {
            if let [column] = spec.columns.as_slice()
                && let Some(parser) = spec
                    .plugin
                    .query_parser(spec.name.clone(), spec.details.as_deref())
            {
                catalog.add_parser(column, parser);
            }
        }
        catalog
    }

    /// Answer expressions on `column` from the memtable's own key index too,
    /// after any maintained index on it.
    pub(crate) fn add_own_key_index(&mut self, column: &str) {
        if let Some(parser) = BTreeMemIndexPlugin.query_parser(OWN_KEY_INDEX.to_string(), None) {
            self.add_parser(column, parser);
        }
    }

    /// The first parser added for a column answers an expression it claims,
    /// as on the base table.
    fn add_parser(&mut self, column: &str, parser: Box<dyn ScalarQueryParser>) {
        if let Some((_, existing)) = self.columns.get_mut(column) {
            existing.add(parser);
            return;
        }
        let Some(field) = self.schema.as_ref().and_then(|schema| schema.field(column)) else {
            return;
        };
        self.columns.insert(
            column.to_string(),
            (field.data_type(), MultiQueryParser::single(parser)),
        );
    }

    /// Whether no index claims expressions on any column.
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

/// A filter split into index searches.
#[derive(Debug)]
pub struct IndexedFilter {
    /// The index searches, combined with `AND` and `OR`.
    pub searches: ScalarIndexExpr,
    /// Whether the searches are the whole filter, with nothing left over.
    pub is_whole_filter: bool,
}

/// Split `filter` into index searches; `None` when no index can help.
pub fn plan_filter(filter: &Expr, catalog: &MemIndexCatalog) -> Result<Option<IndexedFilter>> {
    if catalog.is_empty() {
        return Ok(None);
    }
    let split = apply_scalar_indices(see_through_relabelling(filter, catalog), catalog)?;
    Ok(split
        .scalar_query
        .map(ScalarIndexExpr::optimize)
        .filter(is_evaluable)
        .map(|searches| IndexedFilter {
            searches,
            is_whole_filter: split.refine_expr.is_none(),
        }))
}

/// `filter` with each cast of an indexed column removed when the cast changes
/// only the column's nested field names or metadata.
///
/// Field ids on a memtable's nested fields make comparing a list column with a
/// list literal cast the column, which changes no value but hides the column
/// from the parser. Only the index split looks through such a cast.
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
pub fn evaluate(
    expr: &ScalarIndexExpr,
    indexes: &IndexStore,
    ctx: &SearchContext,
) -> Result<MemSearchResult> {
    match expr {
        ScalarIndexExpr::And(lhs, rhs) => {
            let lhs = evaluate(lhs, indexes, ctx)?;
            if lhs.at_most.is_empty() {
                return Ok(lhs);
            }
            Ok(lhs & evaluate(rhs, indexes, ctx)?)
        }
        ScalarIndexExpr::Or(lhs, rhs) => {
            Ok(evaluate(lhs, indexes, ctx)? | evaluate(rhs, indexes, ctx)?)
        }
        ScalarIndexExpr::Not(_) => {
            debug_assert!(false, "planning refuses NOT, so none reaches here");
            Ok(unknown(ctx))
        }
        ScalarIndexExpr::Query(search) => {
            let index = if search.index_name == OWN_KEY_INDEX {
                indexes.own_key_index()
            } else {
                indexes.get_index(&search.index_name).cloned()
            };
            let Some(index) = index else {
                return Ok(unknown(ctx));
            };
            match index.search(&ScalarQuery(search.query.as_ref()), ctx)? {
                Some(MemMatches::Filter(result)) => {
                    let result = result.truncate_to(ctx.max_visible);
                    // The query matches more than the expression, so these rows
                    // are only candidates.
                    Ok(if search.needs_recheck {
                        MemSearchResult::at_most(result.at_most)
                    } else {
                        result
                    })
                }
                // A ranked answer or none: nothing is ruled out.
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
    use std::sync::Arc;

    use arrow_array::{Int32Array, RecordBatch, StringArray};
    use arrow_schema::{Field, Schema as ArrowSchema};
    use datafusion::common::ScalarValue;
    use lance_datafusion::planner::Planner;
    use lance_index::scalar::SargableQuery;
    use lance_index::scalar::expression::{LabelListQueryParser, ScalarIndexSearch};

    use super::*;

    /// The positions a tree selected, and whether they are the exact answer.
    fn positions(result: MemSearchResult) -> (Vec<u64>, bool) {
        let exact = result.is_exact();
        (result.at_most.into(), exact)
    }

    /// A label-list parser claims a list comparison even though field ids make
    /// the planner cast the column.
    #[test]
    fn an_index_sees_through_a_cast_that_only_relabels_nested_fields() {
        let item = Field::new("item", DataType::Utf8, true).with_metadata(HashMap::from([(
            "lance:field_id".to_string(),
            "3".to_string(),
        )]));
        let tags = DataType::List(Arc::new(item));
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
            schema: None,
        };
        let split = plan_filter(&filter, &catalog).unwrap();
        assert!(split.is_some(), "the label list must claim {filter}");
    }

    fn schema() -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("other", DataType::Int32, true),
        ]))
    }

    /// Two B-trees, on `id` and `name`, over ten rows: `id` counts up and
    /// `name` is `alpha<id>`, with no `_`, which `LIKE` treats as a wildcard.
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

    fn plan(filter: &str, specs: &[MemIndexSpec]) -> Option<IndexedFilter> {
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
        positions(evaluate(&split.searches, &store, &SearchContext::new(u64::MAX)).unwrap())
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

    /// `AND` and `OR` across two indexes compose.
    #[test]
    fn compound_filters_combine_two_indexes() {
        let (positions, exact) = run("id >= 4 AND name = 'alpha5'");
        assert_eq!(positions, vec![5]);
        assert!(exact);

        let (positions, exact) = run("id = 1 OR name = 'alpha7'");
        assert_eq!(positions, vec![1, 7]);
        assert!(exact);
    }

    /// Bounds on one column, split by another conjunct, are searched as one
    /// range: each bound alone is over the budget, together they are not.
    #[test]
    fn split_bounds_on_one_column_search_as_one_range() {
        let (store, specs) = store();
        let split = plan("id >= 2 AND name = 'alpha4' AND id <= 6", &specs).unwrap();
        let ctx = SearchContext::new(9).with_match_budget(5);
        let (positions, exact) = positions(evaluate(&split.searches, &store, &ctx).unwrap());
        assert_eq!(positions, vec![4]);
        assert!(exact, "no bound should have declined");
    }

    /// The memtable's own key index answers a key filter exactly.
    #[test]
    fn the_own_key_index_answers_exactly() {
        let arrow = schema();
        let lance = LanceSchema::try_from(arrow.as_ref()).unwrap();
        let mut store = IndexStore::from_specs(&[], &lance, 1_000, 16).unwrap();
        store.enable_pk_index(&[("id".to_string(), 0)]);
        let ids: Vec<i32> = (0..10).collect();
        let batch = RecordBatch::try_new(
            arrow.clone(),
            vec![
                Arc::new(Int32Array::from(ids.clone())),
                Arc::new(StringArray::from(vec![Some("x"); 10])),
                Arc::new(Int32Array::from(ids)),
            ],
        )
        .unwrap();
        store.insert(&batch, 0).unwrap();

        let planner = Planner::new(arrow);
        let filter = planner
            .optimize_expr(planner.parse_filter("id = 4").unwrap())
            .unwrap();
        let split = plan_filter(&filter, store.filter_catalog())
            .unwrap()
            .expect("the key filter reaches the own key index");
        let found = positions(evaluate(&split.searches, &store, &SearchContext::new(9)).unwrap());
        assert_eq!(found, (vec![4], true));
    }

    /// An unindexed conjunct comes back as the leftover; the indexed one still
    /// narrows.
    #[test]
    fn an_unindexed_conjunct_becomes_the_leftover_expression() {
        let (_, specs) = store();
        let split =
            plan("id >= 4 AND other = 1", &specs).expect("the indexed half reaches an index");
        assert!(
            !split.is_whole_filter,
            "the `other` half has no index and must be left to the filter"
        );
    }

    /// Nothing is indexed, so there is nothing to plan and the caller scans.
    #[test]
    fn a_filter_on_no_indexed_column_declines() {
        let (_, specs) = store();
        assert!(plan("other = 1", &specs).is_none());
    }

    /// A filter with `NOT` over an indexed leaf is scanned.
    #[test]
    fn a_negated_filter_declines_rather_than_answering_wrongly() {
        let (_, specs) = store();
        assert!(plan("NOT (id = 5)", &specs).is_none());
    }

    /// A float zero reaches the index as both zeros, as the full scan reads it.
    #[test]
    fn a_float_zero_reaches_the_index_as_both_zeros() {
        let arrow = Arc::new(ArrowSchema::new(vec![Field::new(
            "value",
            DataType::Float64,
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
            let ScalarIndexExpr::Query(search) = split.searches else {
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

    /// Positions past the visibility watermark are not returned.
    #[test]
    fn evaluation_honors_the_visibility_watermark() {
        let (store, specs) = store();
        let split = plan("id >= 0", &specs).unwrap();
        let (positions, _) =
            positions(evaluate(&split.searches, &store, &SearchContext::new(4)).unwrap());
        assert_eq!(positions, vec![0, 1, 2, 3, 4]);
    }

    /// An index the tree names but the store does not hold rules nothing out.
    #[test]
    fn a_missing_index_rules_nothing_out() {
        let (store, _) = store();
        let missing = ScalarIndexExpr::Query(ScalarIndexSearch {
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
