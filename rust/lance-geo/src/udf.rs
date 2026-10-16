// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use datafusion::arrow::datatypes::DataType;
use datafusion::common::{Result, ScalarValue};
use datafusion::logical_expr::simplify::{ExprSimplifyResult, SimplifyContext};
use datafusion::logical_expr::{
    ColumnarValue, Documentation, Expr, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
};
use datafusion::prelude::SessionContext;
use geodatafusion::udf::geo::relationships::{
    Contains, CoveredBy, Covers, Crosses, Disjoint, Equals, Intersects, Overlaps, Touches, Within,
};

pub fn register_functions(ctx: &SessionContext) {
    let predicates: [(ScalarUDF, ScalarUDF); 10] = [
        (Contains::new().into(), Within::new().into()),
        (Within::new().into(), Contains::new().into()),
        (Covers::new().into(), CoveredBy::new().into()),
        (CoveredBy::new().into(), Covers::new().into()),
        (Intersects::new().into(), Intersects::new().into()),
        (Disjoint::new().into(), Disjoint::new().into()),
        (Touches::new().into(), Touches::new().into()),
        (Crosses::new().into(), Crosses::new().into()),
        (Overlaps::new().into(), Overlaps::new().into()),
        (Equals::new().into(), Equals::new().into()),
    ];
    for (inner, converse) in predicates {
        ctx.register_udf(Relate { inner, converse }.into());
    }
}

/// Keeps literal geometries on the right for both evaluation and index planning.
#[derive(Debug, PartialEq, Eq, Hash)]
struct Relate {
    inner: ScalarUDF,
    converse: ScalarUDF,
}

impl ScalarUDFImpl for Relate {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn signature(&self) -> &Signature {
        self.inner.signature()
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        self.inner.return_type(arg_types)
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.inner.documentation()
    }

    fn invoke_with_args(&self, mut args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        // TODO: Remove these evaluation guards after upgrading to a release with
        // https://github.com/datafusion-contrib/geodatafusion/pull/111.
        if args
            .args
            .iter()
            .any(|arg| matches!(arg, ColumnarValue::Scalar(value) if value.is_null()))
        {
            let null = ScalarValue::Boolean(None);
            return match args.args.iter().find_map(|arg| match arg {
                ColumnarValue::Array(array) => Some(array.len()),
                ColumnarValue::Scalar(_) => None,
            }) {
                Some(len) => Ok(ColumnarValue::Array(null.to_array_of_size(len)?)),
                None => Ok(ColumnarValue::Scalar(null)),
            };
        }

        if matches!(
            args.args.as_slice(),
            [ColumnarValue::Scalar(_), ColumnarValue::Array(_)]
        ) {
            // The upstream scalar-first path computes the relation backwards.
            // Swap the fields too: they carry the GeoArrow extension metadata.
            args.args.swap(0, 1);
            args.arg_fields.swap(0, 1);
            self.converse.invoke_with_args(args)
        } else {
            self.inner.invoke_with_args(args)
        }
    }

    fn simplify(&self, mut args: Vec<Expr>, _info: &SimplifyContext) -> Result<ExprSimplifyResult> {
        if matches!(args.as_slice(), [Expr::Literal(..), right] if !matches!(right, Expr::Literal(..)))
        {
            args.swap(0, 1);
            let converse = ScalarUDF::from(Self {
                inner: self.converse.clone(),
                converse: self.inner.clone(),
            });
            // Keep the wrapper so NULL handling also applies after simplification.
            Ok(ExprSimplifyResult::Simplified(converse.call(args)))
        } else {
            Ok(ExprSimplifyResult::Original(args))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::arrow::array::{Array, record_batch};
    use datafusion::arrow::datatypes::{self as arrow_schema, Field};
    use datafusion::prelude::col;
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::contains("st_contains", "st_within")]
    #[case::within("st_within", "st_contains")]
    #[case::covers("st_covers", "st_coveredby")]
    #[case::covered_by("st_coveredby", "st_covers")]
    #[case::intersects("st_intersects", "st_intersects")]
    #[case::disjoint("st_disjoint", "st_disjoint")]
    #[case::touches("st_touches", "st_touches")]
    #[case::crosses("st_crosses", "st_crosses")]
    #[case::overlaps("st_overlaps", "st_overlaps")]
    #[case::equals("st_equals", "st_equals")]
    #[tokio::test]
    async fn test_relate_argument_order(#[case] name: &str, #[case] converse: &str) {
        let ctx = SessionContext::new();
        crate::register_functions(&ctx);
        let input = record_batch!((
            "wkt",
            Utf8,
            vec![
                Some("POINT (1 1)"),
                Some("POINT (0 2)"),
                Some("POINT (5 5)"),
                Some("LINESTRING (0 2, 4 2)"),
                Some("LINESTRING (2 0, 2 4)"),
                Some("POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))"),
                Some("POLYGON ((2 2, 6 2, 6 6, 2 6, 2 2))"),
                None,
            ]
        ))
        .unwrap();
        ctx.register_batch("t", input).unwrap();
        let batches = ctx
            .sql("SELECT ST_GeomFromText(wkt) AS geom FROM t")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let batch = &batches[0];
        let array = batch["geom"].clone();
        let field = Arc::new(batch.schema().field_with_name("geom").unwrap().clone());
        let state = ctx.state();
        let udf = &state.scalar_functions()[name];
        let invoke = |args: Vec<ColumnarValue>| {
            udf.invoke_with_args(ScalarFunctionArgs {
                args,
                arg_fields: vec![field.clone(); 2],
                number_rows: array.len(),
                return_field: Arc::new(Field::new("result", DataType::Boolean, true)),
                config_options: Arc::new(Default::default()),
            })
            .unwrap()
        };

        // Compare the prepared scalar paths to the correct per-row evaluator,
        // including NULL constants and NULL array elements in either position.
        for wkt in [
            "'POINT (1 1)'",
            "'LINESTRING (0 2, 4 2)'",
            "'POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))'",
            "CAST(NULL AS VARCHAR)",
        ] {
            let constant = ctx
                .sql(&format!("SELECT ST_GeomFromText({wkt}) AS geom"))
                .await
                .unwrap()
                .collect()
                .await
                .unwrap();
            let scalar = ScalarValue::try_from_array(&constant[0]["geom"], 0).unwrap();
            let repeated = scalar.to_array_of_size(array.len()).unwrap();
            for is_scalar_first in [true, false] {
                let mut scalar_args = vec![
                    ColumnarValue::Scalar(scalar.clone()),
                    ColumnarValue::Array(array.clone()),
                ];
                let mut array_args = vec![
                    ColumnarValue::Array(repeated.clone()),
                    ColumnarValue::Array(array.clone()),
                ];
                if !is_scalar_first {
                    scalar_args.swap(0, 1);
                    array_args.swap(0, 1);
                }
                let actual = invoke(scalar_args).into_array(array.len()).unwrap();
                let expected = invoke(array_args).into_array(array.len()).unwrap();
                assert_eq!(
                    actual.as_ref(),
                    expected.as_ref(),
                    "{name}, scalar={wkt}, first={is_scalar_first}"
                );
            }

            let result = invoke(vec![
                ColumnarValue::Scalar(scalar.clone()),
                ColumnarValue::Scalar(scalar.clone()),
            ])
            .into_array(1)
            .unwrap();
            assert_eq!(result.is_null(0), scalar.is_null());
        }

        let scalar = ScalarValue::try_from_array(&array, 5).unwrap();
        let literal = Expr::Literal(scalar, Some(field.metadata().clone().into()));
        let info = SimplifyContext::builder().build();
        let args = vec![literal.clone(), col("geom")];
        let ExprSimplifyResult::Simplified(Expr::ScalarFunction(function)) =
            udf.simplify(args, &info).unwrap()
        else {
            panic!("{name} did not normalize the literal-first expression");
        };
        assert_eq!(function.func.name(), converse);
        assert_eq!(function.args, vec![col("geom"), literal]);
        assert!(function.func.inner().as_ref().is::<Relate>());
    }

    #[rstest]
    #[case::empty(0)]
    #[case::nonempty(3)]
    fn test_null_scalar_result_shape(#[case] len: usize) {
        let udf = Relate {
            inner: Contains::new().into(),
            converse: Within::new().into(),
        };
        let field = Arc::new(Field::new("geom", DataType::Null, true));
        for is_scalar_first in [true, false] {
            let mut args = vec![
                ColumnarValue::Scalar(ScalarValue::Null),
                ColumnarValue::Array(ScalarValue::Null.to_array_of_size(len).unwrap()),
            ];
            if !is_scalar_first {
                args.swap(0, 1);
            }
            let result = udf
                .invoke_with_args(ScalarFunctionArgs {
                    args,
                    arg_fields: vec![field.clone(); 2],
                    number_rows: len,
                    return_field: Arc::new(Field::new("result", DataType::Boolean, true)),
                    config_options: Arc::new(Default::default()),
                })
                .unwrap();
            let ColumnarValue::Array(array) = result else {
                panic!("Expected a NULL array");
            };
            assert_eq!(array.data_type(), &DataType::Boolean);
            assert_eq!(array.len(), len);
            assert_eq!(array.null_count(), len);
        }
    }
}
