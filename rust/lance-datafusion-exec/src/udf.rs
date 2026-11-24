// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Datafusion user defined functions

use arrow_array::{Array, ArrayRef, BooleanArray, StringArray};
use arrow_schema::DataType;
use datafusion_common::DataFusionError;
use datafusion_expr::function::Hint;
use datafusion_expr::{ScalarUDF, Volatility, create_udf};
use datafusion_functions::utils::make_scalar_function;
use std::sync::{Arc, LazyLock};

pub mod json;

/// This method checks whether a string contains all specified tokens. The tokens are separated by
/// punctuations and white spaces.
///
/// The functionality is equivalent to FTS MatchQuery (with fuzziness disabled, Operator::And,
/// and using the simple tokenizer). If FTS index exists and suites the query, it will be used to
/// optimize the query.
///
/// Usage
/// * Use `contains_tokens` in sql.
/// ```rust,ignore
/// let sql = "SELECT * FROM table WHERE contains_tokens(text_col, 'fox jumps dog')";
/// let mut ds = Dataset::open(&ds_path).await?;
/// let ctx = SessionContext::new();
/// ctx.register_table(
///     "table",
///     Arc::new(LanceTableProvider::new(dataset, false, false)),
/// )?;
/// register_functions(&ctx);
/// let df = ctx.sql(sql).await?;
/// ```
fn contains_tokens() -> ScalarUDF {
    let function = Arc::new(make_scalar_function(
        |args: &[ArrayRef]| {
            let column = args[0].as_any().downcast_ref::<StringArray>().ok_or(
                DataFusionError::Execution(
                    "First argument of contains_tokens can't be cast to string".to_string(),
                ),
            )?;
            let queries = args[1].as_any().downcast_ref::<StringArray>().ok_or(
                DataFusionError::Execution(
                    "Second argument of contains_tokens can't be cast to string".to_string(),
                ),
            )?;

            // A literal query arrives as one row (see the hint below) and is
            // tokenized once; a column query is read row by row. A NULL text
            // or query is NULL, not a match: an empty token list would
            // otherwise match every row.
            let literal_tokens = (queries.len() == 1 && queries.is_valid(0))
                .then(|| collect_tokens(queries.value(0)));
            let result = column.iter().enumerate().map(|(i, text)| {
                let text = text?;
                let row = if queries.len() == 1 { 0 } else { i };
                if queries.is_null(row) {
                    return None;
                }
                let text_tokens = collect_tokens(text);
                let matches = |tokens: &[&str]| tokens.iter().all(|t| text_tokens.contains(t));
                Some(match &literal_tokens {
                    Some(tokens) => matches(tokens),
                    None => matches(&collect_tokens(queries.value(row))),
                })
            });

            Ok(Arc::new(BooleanArray::from_iter(result)) as ArrayRef)
        },
        vec![Hint::Pad, Hint::AcceptsSingular],
    ));

    create_udf(
        "contains_tokens",
        vec![DataType::Utf8, DataType::Utf8],
        DataType::Boolean,
        Volatility::Immutable,
        function,
    )
}

/// Split tokens separated by punctuations and white spaces.
fn collect_tokens(text: &str) -> Vec<&str> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .collect()
}

pub static CONTAINS_TOKENS_UDF: LazyLock<ScalarUDF> = LazyLock::new(contains_tokens);

#[cfg(test)]
mod tests {
    use crate::udf::CONTAINS_TOKENS_UDF;
    use arrow_array::{Array, BooleanArray, StringArray};
    use arrow_schema::{DataType, Field};
    use datafusion_common::ScalarValue;
    use datafusion_expr::{ColumnarValue, ScalarFunctionArgs};
    use std::sync::Arc;

    #[tokio::test]
    async fn test_contains_tokens() {
        // Prepare arguments
        let contains_tokens = CONTAINS_TOKENS_UDF.clone();
        let text_col = Arc::new(StringArray::from(vec![
            "a cat catch a fish",
            "a fish catch a cat",
            "a white cat catch a big fish",
            "cat catchup fish",
            "cat fish catch",
        ]));
        let token = Arc::new(StringArray::from(vec![
            " cat catch fish.",
            " cat catch fish.",
            " cat catch fish.",
            " cat catch fish.",
            " cat catch fish.",
        ]));

        let args = vec![ColumnarValue::Array(text_col), ColumnarValue::Array(token)];
        let arg_fields = vec![
            Arc::new(Field::new("text_col".to_string(), DataType::Utf8, false)),
            Arc::new(Field::new("token".to_string(), DataType::Utf8, false)),
        ];

        let args = ScalarFunctionArgs {
            args,
            arg_fields,
            number_rows: 5,
            return_field: Arc::new(Field::new("res".to_string(), DataType::Boolean, false)),
            config_options: Arc::new(Default::default()),
        };

        // Invoke contains_tokens manually
        let values = contains_tokens.invoke_with_args(args).unwrap();

        if let ColumnarValue::Array(array) = values {
            let array = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            assert_eq!(
                array.clone(),
                BooleanArray::from(vec![true, true, true, false, true])
            );
        } else {
            panic!("Expected an Array but got {:?}", values);
        }
    }

    fn invoke_contains_tokens(query: ColumnarValue) -> BooleanArray {
        let text_col = Arc::new(StringArray::from(vec![
            Some("a cat catch a fish"),
            Some("dog runs"),
            Some("dog runs"),
            None,
        ]));
        let args = ScalarFunctionArgs {
            args: vec![ColumnarValue::Array(text_col), query],
            arg_fields: vec![
                Arc::new(Field::new("text_col".to_string(), DataType::Utf8, true)),
                Arc::new(Field::new("query".to_string(), DataType::Utf8, true)),
            ],
            number_rows: 4,
            return_field: Arc::new(Field::new("res".to_string(), DataType::Boolean, true)),
            config_options: Arc::new(Default::default()),
        };
        let ColumnarValue::Array(array) = CONTAINS_TOKENS_UDF.invoke_with_args(args).unwrap()
        else {
            panic!("Expected an Array");
        };
        array
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap()
            .clone()
    }

    #[test]
    fn test_contains_tokens_null_query_matches_nothing() {
        // The tokens of a NULL query used to be read as "", an empty token
        // list that every text contains.
        let result = invoke_contains_tokens(ColumnarValue::Scalar(ScalarValue::Utf8(None)));
        assert_eq!(result, BooleanArray::from(vec![None, None, None, None]));
    }

    #[test]
    fn test_contains_tokens_reads_a_column_query_per_row() {
        // Only the first row's query used to be read, for every row, so the
        // second row was matched against "fish". A NULL query is NULL on its
        // own row only.
        let queries = Arc::new(StringArray::from(vec![
            Some("fish"),
            Some("dog"),
            None,
            Some("cat"),
        ]));
        let result = invoke_contains_tokens(ColumnarValue::Array(queries));
        assert_eq!(
            result,
            BooleanArray::from(vec![Some(true), Some(true), None, None])
        );
    }
}
