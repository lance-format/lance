// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! UDFs that Lance registers in place of geodatafusion's.
//!
//! The spatial relationship predicates (`ST_Contains`, `ST_Within`, ...) are
//! replaced so that they give the same answer whichever argument is the
//! constant geometry. geodatafusion evaluates `f(<constant>, <array>)` as `relate(array, constant)`,
//! the transpose of the DE-9IM matrix the predicate reads, so `ST_Contains`,
//! `ST_Within`, `ST_Covers` and `ST_CoveredBy` answer the converse question when
//! the constant comes first, and a NULL constant geometry panics. Each predicate
//! is registered here as a wrapper around geodatafusion's implementation that
//!
//! * evaluates a constant-first call as the converse predicate with the arguments
//!   swapped, which goes through geodatafusion's correct array-scalar path;
//! * returns NULL for a NULL constant geometry;
//! * rewrites a constant-first call to its column-first converse during
//!   simplification, which is the form the R-tree index can serve.
//!
//! TODO: once Lance pins a geodatafusion release that includes
//! <https://github.com/datafusion-contrib/geodatafusion/pull/111>, remove the
//! argument swap and NULL handling in `invoke_with_args`. The `simplify` rewrite
//! does not depend on that bug and keeps constant-first predicates indexable.

use std::sync::Arc;

use datafusion::arrow::array::BooleanArray;
use datafusion::arrow::datatypes::DataType;
use datafusion::common::{Result, ScalarValue};
use datafusion::logical_expr::expr::ScalarFunction;
use datafusion::logical_expr::simplify::{ExprSimplifyResult, SimplifyContext};
use datafusion::logical_expr::{
    ColumnarValue, Documentation, Expr, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
};
use datafusion::prelude::SessionContext;
use geodatafusion::udf::geo::relationships as geodatafusion_relate;

/// A DE-9IM relationship between two geometries, as tested by one `ST_*` predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Relation {
    Contains,
    CoveredBy,
    Covers,
    Crosses,
    Disjoint,
    Equals,
    Intersects,
    Overlaps,
    Touches,
    Within,
}

impl Relation {
    const ALL: [Self; 10] = [
        Self::Contains,
        Self::CoveredBy,
        Self::Covers,
        Self::Crosses,
        Self::Disjoint,
        Self::Equals,
        Self::Intersects,
        Self::Overlaps,
        Self::Touches,
        Self::Within,
    ];

    /// The relation that holds for `(b, a)` exactly when `self` holds for `(a, b)`.
    ///
    /// Swapping the arguments transposes the DE-9IM matrix, which turns contains
    /// into within and covers into covered-by. The other predicates are symmetric.
    fn converse(self) -> Self {
        match self {
            Self::Contains => Self::Within,
            Self::Within => Self::Contains,
            Self::Covers => Self::CoveredBy,
            Self::CoveredBy => Self::Covers,
            Self::Crosses
            | Self::Disjoint
            | Self::Equals
            | Self::Intersects
            | Self::Overlaps
            | Self::Touches => self,
        }
    }

    fn geodatafusion_udf(self) -> ScalarUDF {
        match self {
            Self::Contains => geodatafusion_relate::Contains::new().into(),
            Self::CoveredBy => geodatafusion_relate::CoveredBy::new().into(),
            Self::Covers => geodatafusion_relate::Covers::new().into(),
            Self::Crosses => geodatafusion_relate::Crosses::new().into(),
            Self::Disjoint => geodatafusion_relate::Disjoint::new().into(),
            Self::Equals => geodatafusion_relate::Equals::new().into(),
            Self::Intersects => geodatafusion_relate::Intersects::new().into(),
            Self::Overlaps => geodatafusion_relate::Overlaps::new().into(),
            Self::Touches => geodatafusion_relate::Touches::new().into(),
            Self::Within => geodatafusion_relate::Within::new().into(),
        }
    }
}

/// A geodatafusion relate predicate that is evaluated correctly when the
/// constant geometry is the first argument.
#[derive(Debug, PartialEq, Eq, Hash)]
struct RelateUdf {
    relation: Relation,
    /// geodatafusion's implementation of `relation`.
    inner: ScalarUDF,
    /// geodatafusion's implementation of the converse relation, which evaluates
    /// a constant-first call with the arguments swapped.
    converse: ScalarUDF,
}

impl RelateUdf {
    fn new(relation: Relation) -> Self {
        Self {
            relation,
            inner: relation.geodatafusion_udf(),
            converse: relation.converse().geodatafusion_udf(),
        }
    }
}

impl ScalarUDFImpl for RelateUdf {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn signature(&self) -> &Signature {
        self.inner.signature()
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        self.inner.return_type(arg_types)
    }

    fn invoke_with_args(&self, mut args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        // Relating anything to a NULL geometry is NULL, as in PostGIS.
        if args
            .args
            .iter()
            .any(|arg| matches!(arg, ColumnarValue::Scalar(value) if value.is_null()))
        {
            return Ok(match args.args.as_slice() {
                [ColumnarValue::Scalar(_), ColumnarValue::Scalar(_)] => {
                    ColumnarValue::Scalar(ScalarValue::Boolean(None))
                }
                _ => ColumnarValue::Array(Arc::new(BooleanArray::new_null(args.number_rows))),
            });
        }
        match args.args.as_slice() {
            // geodatafusion relates a constant first argument as if it were the
            // second, so put the constant second and test the converse relation.
            [ColumnarValue::Scalar(_), ColumnarValue::Array(_)] => {
                args.args.swap(0, 1);
                args.arg_fields.swap(0, 1);
                self.converse.invoke_with_args(args)
            }
            _ => self.inner.invoke_with_args(args),
        }
    }

    /// Rewrites `f(<literal>, x)` to `converse(f)(x, <literal>)`.
    ///
    /// The R-tree index only recognises spatial predicates whose first argument
    /// is the indexed column, so this lets it serve constant-first predicates.
    fn simplify(&self, args: Vec<Expr>, _info: &SimplifyContext) -> Result<ExprSimplifyResult> {
        let [first, second] = match <[Expr; 2]>::try_from(args) {
            Ok(args) => args,
            Err(args) => return Ok(ExprSimplifyResult::Original(args)),
        };
        if !matches!(first, Expr::Literal(..)) || matches!(second, Expr::Literal(..)) {
            return Ok(ExprSimplifyResult::Original(vec![first, second]));
        }
        let converse = ScalarUDF::new_from_impl(Self::new(self.relation.converse()));
        Ok(ExprSimplifyResult::Simplified(Expr::ScalarFunction(
            ScalarFunction::new_udf(Arc::new(converse), vec![second, first]),
        )))
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.inner.documentation()
    }
}

/// Register Lance's UDFs, replacing geodatafusion's registrations of the same
/// names. Call this after `geodatafusion::register`.
pub fn register_functions(ctx: &SessionContext) {
    for relation in Relation::ALL {
        ctx.register_udf(ScalarUDF::new_from_impl(RelateUdf::new(relation)));
    }
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::array::{ArrayRef, AsArray};
    use datafusion::arrow::datatypes::{Field, FieldRef};
    use datafusion::config::ConfigOptions;
    use datafusion::execution::FunctionRegistry;
    use datafusion::prelude::{col, lit};
    use geo_types::{Geometry, wkt};
    use geoarrow_array::GeoArrowArray;
    use geoarrow_array::builder::WkbBuilder;
    use geoarrow_schema::WkbType;

    use super::*;

    fn square() -> Geometry {
        wkt! { POLYGON((0.0 0.0, 4.0 0.0, 4.0 4.0, 0.0 4.0, 0.0 0.0)) }.into()
    }

    /// Geometries of every dimension placed around [`square`], so that each
    /// relation holds for some pairs and not for others.
    fn geometries() -> Vec<Option<Geometry>> {
        vec![
            Some(wkt! { POINT(1.0 1.0) }.into()),
            Some(wkt! { POINT(4.0 2.0) }.into()),
            Some(wkt! { POINT(5.0 5.0) }.into()),
            Some(wkt! { LINESTRING(1.0 1.0, 3.0 3.0) }.into()),
            Some(wkt! { LINESTRING(0.0 0.0, 4.0 0.0) }.into()),
            Some(wkt! { LINESTRING(-1.0 2.0, 5.0 2.0) }.into()),
            Some(wkt! { LINESTRING(0.0 4.0, 4.0 0.0) }.into()),
            Some(wkt! { POLYGON((1.0 1.0, 3.0 1.0, 3.0 3.0, 1.0 3.0, 1.0 1.0)) }.into()),
            Some(wkt! { POLYGON((2.0 2.0, 6.0 2.0, 6.0 6.0, 2.0 6.0, 2.0 2.0)) }.into()),
            Some(wkt! { POLYGON((-1.0 -1.0, 5.0 -1.0, 5.0 5.0, -1.0 5.0, -1.0 -1.0)) }.into()),
            Some(square()),
            Some(wkt! { POLYGON((4.0 0.0, 8.0 0.0, 8.0 4.0, 4.0 4.0, 4.0 0.0)) }.into()),
            None,
        ]
    }

    fn wkb_array(geometries: &[Option<Geometry>]) -> (ArrayRef, FieldRef) {
        let array = WkbBuilder::<i32>::from_nullable_geometries(geometries, WkbType::default())
            .unwrap()
            .finish();
        let field = Arc::new(array.data_type().to_field("geom", true));
        (array.to_array_ref(), field)
    }

    fn array_arg(geometries: &[Option<Geometry>]) -> (ColumnarValue, FieldRef) {
        let (array, field) = wkb_array(geometries);
        (ColumnarValue::Array(array), field)
    }

    fn scalar(geometry: Option<Geometry>) -> (ScalarValue, FieldRef) {
        let (array, field) = wkb_array(&[geometry]);
        (ScalarValue::try_from_array(&array, 0).unwrap(), field)
    }

    fn scalar_arg(geometry: Option<Geometry>) -> (ColumnarValue, FieldRef) {
        let (scalar, field) = scalar(geometry);
        (ColumnarValue::Scalar(scalar), field)
    }

    fn invoke(
        udf: &ScalarUDF,
        args: [(ColumnarValue, FieldRef); 2],
        number_rows: usize,
    ) -> Vec<Option<bool>> {
        let (args, arg_fields) = args.into_iter().unzip();
        let result = udf
            .invoke_with_args(ScalarFunctionArgs {
                args,
                arg_fields,
                number_rows,
                return_field: Arc::new(Field::new(udf.name(), DataType::Boolean, true)),
                config_options: Arc::new(ConfigOptions::default()),
            })
            .unwrap();
        result
            .into_array(number_rows)
            .unwrap()
            .as_boolean()
            .iter()
            .collect()
    }

    fn lance_udf(relation: Relation) -> ScalarUDF {
        ScalarUDF::new_from_impl(RelateUdf::new(relation))
    }

    /// With a constant in either position, every predicate must agree with
    /// geodatafusion's array-array evaluation, which relates each pair of rows
    /// with `geo::Relate` in argument order and is not affected by the bug.
    #[test]
    fn test_constant_in_either_position_matches_row_by_row_relate() {
        let rows = geometries();
        let num_rows = rows.len();
        for relation in Relation::ALL {
            let udf = lance_udf(relation);
            let converse = lance_udf(relation.converse());
            let reference = relation.geodatafusion_udf();
            for constant in rows.iter().flatten() {
                let repeated = vec![Some(constant.clone()); num_rows];

                let constant_first = invoke(
                    &udf,
                    [scalar_arg(Some(constant.clone())), array_arg(&rows)],
                    num_rows,
                );
                let expected = invoke(
                    &reference,
                    [array_arg(&repeated), array_arg(&rows)],
                    num_rows,
                );
                assert_eq!(constant_first, expected, "{relation:?}({constant:?}, rows)");

                let constant_second = invoke(
                    &udf,
                    [array_arg(&rows), scalar_arg(Some(constant.clone()))],
                    num_rows,
                );
                let expected = invoke(
                    &reference,
                    [array_arg(&rows), array_arg(&repeated)],
                    num_rows,
                );
                assert_eq!(
                    constant_second, expected,
                    "{relation:?}(rows, {constant:?})"
                );

                let column_first_converse = invoke(
                    &converse,
                    [array_arg(&rows), scalar_arg(Some(constant.clone()))],
                    num_rows,
                );
                assert_eq!(
                    constant_first, column_first_converse,
                    "{relation:?}({constant:?}, rows) vs its converse"
                );
            }
        }
    }

    /// Spot-check the asymmetric predicates against known answers, so the
    /// comparison above cannot pass by agreeing on a wrong reference.
    #[test]
    fn test_asymmetric_predicates_with_constant_first() {
        let points = vec![
            Some(wkt! { POINT(1.0 1.0) }.into()),
            Some(wkt! { POINT(4.0 2.0) }.into()),
            Some(wkt! { POINT(5.0 5.0) }.into()),
        ];
        // (relation, expected for the inside, boundary and outside points)
        let cases = [
            (Relation::Contains, [true, false, false]),
            (Relation::Covers, [true, true, false]),
            (Relation::Within, [false, false, false]),
            (Relation::CoveredBy, [false, false, false]),
        ];
        for (relation, expected) in cases {
            let actual = invoke(
                &lance_udf(relation),
                [scalar_arg(Some(square())), array_arg(&points)],
                points.len(),
            );
            assert_eq!(actual, expected.map(Some), "{relation:?}(square, points)");
        }
    }

    #[test]
    fn test_null_constant_returns_null() {
        // An empty batch must still get an empty array back.
        for rows in [geometries(), vec![]] {
            let num_rows = rows.len();
            for relation in Relation::ALL {
                let udf = lance_udf(relation);
                for null in [
                    scalar_arg(None),
                    (
                        ColumnarValue::Scalar(ScalarValue::Null),
                        Arc::new(Field::new("geom", DataType::Null, true)),
                    ),
                ] {
                    assert_eq!(
                        invoke(&udf, [null.clone(), array_arg(&rows)], num_rows),
                        vec![None; num_rows],
                        "{relation:?}(NULL, {num_rows} rows)"
                    );
                    assert_eq!(
                        invoke(&udf, [array_arg(&rows), null.clone()], num_rows),
                        vec![None; num_rows],
                        "{relation:?}({num_rows} rows, NULL)"
                    );
                    assert_eq!(
                        invoke(&udf, [null.clone(), scalar_arg(Some(square()))], 1),
                        vec![None],
                        "{relation:?}(NULL, square)"
                    );
                }
            }
        }
    }

    #[test]
    fn test_simplify_puts_the_constant_second() {
        let (constant, _) = scalar(Some(square()));
        let point = Expr::ScalarFunction(ScalarFunction::new_udf(
            Arc::new(geodatafusion::udf::native::constructors::Point::default().into()),
            vec![col("x"), col("y")],
        ));
        let simplify = |relation: Relation, args: Vec<Expr>| {
            lance_udf(relation)
                .simplify(args, &SimplifyContext::default())
                .unwrap()
        };
        for relation in Relation::ALL {
            let converse = Arc::new(lance_udf(relation.converse()));
            for other in [col("geom"), point.clone()] {
                let ExprSimplifyResult::Simplified(simplified) =
                    simplify(relation, vec![lit(constant.clone()), other.clone()])
                else {
                    panic!("{relation:?}(<constant>, {other}) was not rewritten");
                };
                assert_eq!(
                    simplified,
                    Expr::ScalarFunction(ScalarFunction::new_udf(
                        converse.clone(),
                        vec![other.clone(), lit(constant.clone())],
                    )),
                );
                assert!(matches!(
                    simplify(
                        relation.converse(),
                        vec![other.clone(), lit(constant.clone())]
                    ),
                    ExprSimplifyResult::Original(_)
                ));
            }
            assert!(matches!(
                simplify(relation, vec![lit(constant.clone()), lit(constant.clone())]),
                ExprSimplifyResult::Original(_)
            ));
        }
    }

    #[test]
    fn test_register_replaces_geodatafusion_predicates() {
        let ctx = SessionContext::new();
        crate::register_functions(&ctx);
        for relation in Relation::ALL {
            let udf = lance_udf(relation);
            assert_eq!(ctx.udf(udf.name()).unwrap().as_ref(), &udf);
        }
    }
}
