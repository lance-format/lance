// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Spatial queries and the parser that finds them in filter expressions.

use std::any::Any;
use std::ops::Bound;
use std::sync::Arc;

use arrow_schema::{DataType, Field};
use datafusion_common::ScalarValue;
use datafusion_expr::{Expr, Operator, ScalarUDF};
use lance_index::scalar::AnyQuery;
use lance_index::scalar::expression::{IndexedExpression, ScalarQueryParser};

#[derive(Debug, Clone, PartialEq)]
pub struct RelationQuery {
    pub value: ScalarValue,
    pub field: Field,
}

/// A query that a Geo index can satisfy
#[derive(Debug, Clone, PartialEq)]
pub enum GeoQuery {
    IntersectQuery(RelationQuery),
    IsNull,
}

impl AnyQuery for GeoQuery {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn format(&self, col: &str) -> String {
        match self {
            Self::IntersectQuery(query) => {
                format!("Intersect({} {})", col, query.value)
            }
            Self::IsNull => {
                format!("{} IS NULL", col)
            }
        }
    }

    fn to_expr(&self, _col: String) -> Expr {
        todo!()
    }

    fn dyn_eq(&self, other: &dyn AnyQuery) -> bool {
        match other.as_any().downcast_ref::<Self>() {
            Some(o) => self == o,
            None => false,
        }
    }
}

/// A parser for geo indices that handles spatial queries
#[derive(Debug, Clone)]
pub struct GeoQueryParser {
    index_name: String,
    index_type: String,
}

impl GeoQueryParser {
    pub fn new(index_name: String, index_type: String) -> Self {
        Self {
            index_name,
            index_type,
        }
    }
}

impl ScalarQueryParser for GeoQueryParser {
    fn visit_between(
        &self,
        _: &str,
        _: &Bound<ScalarValue>,
        _: &Bound<ScalarValue>,
    ) -> Option<IndexedExpression> {
        None
    }

    fn visit_in_list(&self, _: &str, _: &[ScalarValue]) -> Option<IndexedExpression> {
        None
    }

    fn visit_is_bool(&self, _: &str, _: bool) -> Option<IndexedExpression> {
        None
    }

    fn visit_is_null(&self, column: &str) -> Option<IndexedExpression> {
        Some(IndexedExpression::index_query_with_recheck(
            column.to_string(),
            self.index_name.clone(),
            self.index_type.clone(),
            Arc::new(GeoQuery::IsNull),
            true,
        ))
    }

    fn visit_comparison(
        &self,
        _: &str,
        _: &ScalarValue,
        _: &Operator,
    ) -> Option<IndexedExpression> {
        None
    }

    fn visit_scalar_function(
        &self,
        column: &str,
        _data_type: &DataType,
        func: &ScalarUDF,
        args: &[Expr],
    ) -> Option<IndexedExpression> {
        if (func.name() == "st_intersects"
            || func.name() == "st_contains"
            || func.name() == "st_within"
            || func.name() == "st_touches"
            || func.name() == "st_crosses"
            || func.name() == "st_overlaps"
            || func.name() == "st_covers"
            || func.name() == "st_coveredby")
            && args.len() == 2
        {
            let left_arg = &args[0];
            let right_arg = &args[1];
            return match (left_arg, right_arg) {
                (Expr::Literal(left_value, metadata), Expr::Column(_)) => {
                    let mut field = Field::new("_geo", left_value.data_type(), false);
                    if let Some(metadata) = metadata {
                        field = field.with_metadata(metadata.to_hashmap());
                    }
                    let query = GeoQuery::IntersectQuery(RelationQuery {
                        value: left_value.clone(),
                        field,
                    });
                    Some(IndexedExpression::index_query_with_recheck(
                        column.to_string(),
                        self.index_name.clone(),
                        self.index_type.clone(),
                        Arc::new(query),
                        true,
                    ))
                }
                (Expr::Column(_), Expr::Literal(right_value, metadata)) => {
                    let mut field = Field::new("_geo", right_value.data_type(), false);
                    if let Some(metadata) = metadata {
                        field = field.with_metadata(metadata.to_hashmap());
                    }
                    let query = GeoQuery::IntersectQuery(RelationQuery {
                        value: right_value.clone(),
                        field,
                    });
                    Some(IndexedExpression::index_query_with_recheck(
                        column.to_string(),
                        self.index_name.clone(),
                        self.index_type.clone(),
                        Arc::new(query),
                        true,
                    ))
                }
                _ => None,
            };
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use datafusion_common::metadata::FieldMetadata;
    use datafusion_expr::{ColumnarValue, Volatility, col, create_udf};
    use lance_index::scalar::expression::{ScalarIndexExpr, ScalarIndexSearch};
    use rstest::rstest;

    use super::*;

    const EXTENSION_NAME: &str = "ARROW:extension:name";

    fn parser() -> GeoQueryParser {
        GeoQueryParser::new("geom_idx".to_string(), "RTree".to_string())
    }

    fn visit(function: &str, args: &[Expr]) -> Option<IndexedExpression> {
        let udf = create_udf(
            function,
            vec![DataType::Binary, DataType::Binary],
            DataType::Boolean,
            Volatility::Immutable,
            Arc::new(
                |_: &[ColumnarValue]| -> datafusion_common::Result<ColumnarValue> {
                    unreachable!("the parser never runs the function")
                },
            ),
        );
        parser().visit_scalar_function("geom", &DataType::Binary, &udf, args)
    }

    fn geometry() -> Expr {
        Expr::Literal(ScalarValue::Binary(Some(vec![1, 2, 3])), None)
    }

    /// The parser asks for a recheck on every query. Intersect queries depend on it:
    /// the R-tree answers them from bounding boxes, so without it they would return
    /// rows that do not match.
    fn rechecked_search(expression: IndexedExpression) -> ScalarIndexSearch {
        assert!(expression.refine_expr.is_none());
        let Some(ScalarIndexExpr::Query(search)) = expression.scalar_query else {
            panic!("expected a single index query");
        };
        assert_eq!(search.column, "geom");
        assert_eq!(search.index_name, "geom_idx");
        assert_eq!(search.index_type, "RTree");
        assert!(search.needs_recheck);
        search
    }

    #[rstest]
    fn test_spatial_predicates_query_the_index(
        #[values(
            "st_intersects",
            "st_contains",
            "st_within",
            "st_touches",
            "st_crosses",
            "st_overlaps",
            "st_covers",
            "st_coveredby"
        )]
        function: &str,
        #[values(false, true)] is_literal_first: bool,
    ) {
        let value = ScalarValue::Binary(Some(vec![1, 2, 3]));
        let metadata = FieldMetadata::from(HashMap::from([(
            EXTENSION_NAME.to_string(),
            "geoarrow.wkb".to_string(),
        )]));
        let literal = Expr::Literal(value.clone(), Some(metadata));
        let args = if is_literal_first {
            [literal, col("geom")]
        } else {
            [col("geom"), literal]
        };

        let search = rechecked_search(visit(function, &args).unwrap());
        let query = search.query.as_any().downcast_ref::<GeoQuery>().unwrap();
        let GeoQuery::IntersectQuery(relation) = query else {
            panic!("expected an intersect query, got {query:?}");
        };
        assert_eq!(relation.value, value);
        assert_eq!(relation.field.data_type(), &DataType::Binary);
        assert_eq!(relation.field.metadata()[EXTENSION_NAME], "geoarrow.wkb");
    }

    #[test]
    fn test_is_null_queries_the_index() {
        let search = rechecked_search(parser().visit_is_null("geom").unwrap());
        let query = search.query.as_any().downcast_ref::<GeoQuery>().unwrap();
        assert_eq!(query, &GeoQuery::IsNull);
    }

    #[rstest]
    #[case::other_function("st_distance", vec![col("geom"), geometry()])]
    #[case::one_argument("st_intersects", vec![col("geom")])]
    #[case::three_arguments("st_intersects", vec![col("geom"), geometry(), geometry()])]
    #[case::two_columns("st_intersects", vec![col("geom"), col("other")])]
    #[case::two_literals("st_intersects", vec![geometry(), geometry()])]
    fn test_other_calls_are_left_to_the_filter(#[case] function: &str, #[case] args: Vec<Expr>) {
        assert!(visit(function, &args).is_none());
    }
}
