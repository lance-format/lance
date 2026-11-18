// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::DocumentGranularity;
use super::query::{
    BooleanQuery, BoostQuery, CombinedFieldsQuery, FtsQuery, MatchQuery, MultiMatchQuery, Occur,
    Operator, PhraseQuery,
};
use lance_core::{Error, Result};
use serde_json::Value;

pub trait JsonParser {
    fn from_json(value: &Value) -> Result<Self>
    where
        Self: Sized;
}

impl JsonParser for MatchQuery {
    fn from_json(value: &Value) -> Result<Self> {
        let column = value["column"].as_str().map(String::from);
        let terms = value["terms"]
            .as_str()
            .ok_or_else(|| Error::invalid_input("missing terms in match query"))?
            .to_string();
        let boost = value["boost"]
            .as_f64()
            .map(|v| v as f32)
            .unwrap_or(Self::default_boost());
        let fuzziness = match &value["fuzziness"] {
            Value::String(s) if s.eq_ignore_ascii_case("auto") => None,
            Value::String(s) => {
                return Err(Error::invalid_input(format!(
                    "fuzziness must be a non-negative integer or \"auto\", got \"{s}\""
                )));
            }
            other => parse_bounded_u64(other, "fuzziness", u32::MAX as u64)?.map(|v| v as u32),
        };
        let max_expansions = parse_bounded_u64(
            &value["max_expansions"],
            "max_expansions",
            usize::MAX as u64,
        )?
        .map(|v| v as usize)
        .unwrap_or(Self::default_max_expansions());
        let operator = value["operator"]
            .as_str()
            .map(Operator::try_from)
            .transpose()?
            .unwrap_or_default();
        let prefix_length =
            parse_bounded_u64(&value["prefix_length"], "prefix_length", u32::MAX as u64)?
                .map(|v| v as u32)
                .unwrap_or(0);
        let document_granularity = parse_document_granularity(value)?;

        Ok(Self {
            column,
            terms,
            boost,
            fuzziness,
            max_expansions,
            operator,
            prefix_length,
            document_granularity,
        })
    }
}

impl JsonParser for PhraseQuery {
    fn from_json(value: &Value) -> Result<Self> {
        let column = value["column"].as_str().map(String::from);
        let terms = value["terms"]
            .as_str()
            .ok_or_else(|| Error::invalid_input("missing terms in phrase query"))?
            .to_string();
        let slop = parse_bounded_u64(&value["slop"], "slop", u32::MAX as u64)?
            .map(|v| v as u32)
            .unwrap_or(0);
        let document_granularity = parse_document_granularity(value)?;

        Ok(Self {
            column,
            terms,
            slop,
            document_granularity,
        })
    }
}

fn parse_document_granularity(value: &Value) -> Result<Option<DocumentGranularity>> {
    match value.get("document_granularity") {
        None | Some(Value::Null) => Ok(None),
        Some(granularity) => serde_json::from_value(granularity.clone())
            .map(Some)
            .map_err(|error| {
                Error::invalid_input(format!(
                    "invalid document_granularity in FTS query: {error}"
                ))
            }),
    }
}

/// Parses an optional non-negative integer field bounded by `max`, rejecting
/// present-but-invalid values (negative numbers, numbers written as floats
/// such as `1.0`, non-numbers, or values above `max`) instead of silently
/// coercing them to a default.
/// Absent keys and explicit nulls map to `None` so callers keep their defaults.
fn parse_bounded_u64(value: &Value, field: &str, max: u64) -> Result<Option<u64>> {
    let invalid = |repr: &dyn std::fmt::Display| {
        Error::invalid_input(format!(
            "{field} must be a non-negative integer no greater than {max}, got {repr}"
        ))
    };
    match value {
        Value::Null => Ok(None),
        Value::Number(num) => match num.as_u64() {
            Some(raw) if raw <= max => Ok(Some(raw)),
            Some(raw) => Err(invalid(&raw)),
            None => Err(invalid(num)),
        },
        other => Err(invalid(other)),
    }
}

impl JsonParser for BoostQuery {
    fn from_json(value: &Value) -> Result<Self> {
        let positive = value["positive"]
            .as_object()
            .ok_or_else(|| Error::invalid_input("missing positive in boost query"))?;
        let positive_query = from_json_value(&Value::Object(positive.clone()))?;

        let negative = value["negative"]
            .as_object()
            .ok_or_else(|| Error::invalid_input("missing negative in boost query"))?;
        let negative_query = from_json_value(&Value::Object(negative.clone()))?;

        let negative_boost = value["negative_boost"].as_f64().map(|v| v as f32);

        Ok(Self::new(positive_query, negative_query, negative_boost))
    }
}

impl JsonParser for MultiMatchQuery {
    fn from_json(value: &Value) -> Result<Self> {
        let query = value["match_queries"]
            .as_array()
            .ok_or_else(|| Error::invalid_input("missing match_queries in multi_match query"))?;
        let query = query
            .iter()
            .map(MatchQuery::from_json)
            .collect::<Result<Vec<MatchQuery>>>()?;

        if query.is_empty() {
            return Err(Error::invalid_input("empty multi_match query"));
        }

        Ok(Self {
            match_queries: query,
        })
    }
}

impl JsonParser for CombinedFieldsQuery {
    fn from_json(value: &Value) -> Result<Self> {
        let terms = value["query"]
            .as_str()
            .ok_or_else(|| Error::invalid_input("missing query in combined_fields query"))?
            .to_string();
        let columns = value["columns"]
            .as_array()
            .ok_or_else(|| Error::invalid_input("missing columns in combined_fields query"))?
            .iter()
            .map(|v| {
                v.as_str().map(String::from).ok_or_else(|| {
                    Error::invalid_input(
                        "columns must be an array of strings in combined_fields query",
                    )
                })
            })
            .collect::<Result<Vec<String>>>()?;

        let query = Self::try_new(terms, columns)?;

        let query = match value.get("boost") {
            Some(Value::Array(boosts)) => {
                let boosts = boosts
                    .iter()
                    .map(|v| {
                        v.as_f64().map(|f| f as f32).ok_or_else(|| {
                            Error::invalid_input(
                                "boost must be an array of numbers in combined_fields query",
                            )
                        })
                    })
                    .collect::<Result<Vec<f32>>>()?;
                query.try_with_boosts(boosts)?
            }
            _ => query,
        };

        let operator = value["operator"]
            .as_str()
            .map(Operator::try_from)
            .transpose()?
            .unwrap_or_default();

        Ok(query.with_operator(operator))
    }
}

impl JsonParser for BooleanQuery {
    fn from_json(value: &Value) -> Result<Self> {
        let mut clauses = Vec::new();

        if let Some(must) = value["must"].as_array() {
            for query_val in must {
                let query = from_json_value(query_val)?;
                clauses.push((Occur::Must, query));
            }
        }

        if let Some(should) = value["should"].as_array() {
            for query_val in should {
                let query = from_json_value(query_val)?;
                clauses.push((Occur::Should, query));
            }
        }

        if let Some(must_not) = value["must_not"].as_array() {
            for query_val in must_not {
                let query = from_json_value(query_val)?;
                clauses.push((Occur::MustNot, query));
            }
        }

        Ok(Self::new(clauses))
    }
}

fn from_json_value(value: &Value) -> Result<FtsQuery> {
    let value = value
        .as_object()
        .ok_or_else(|| Error::invalid_input("value must be a JSON object"))?;
    if value.len() != 1 {
        return Err(Error::invalid_input("value must be a single JSON object"));
    }

    let (query_type, query_val) = value.into_iter().next().unwrap();
    match query_type.as_str() {
        "match" => Ok(FtsQuery::Match(MatchQuery::from_json(query_val)?)),
        "phrase" => Ok(FtsQuery::Phrase(PhraseQuery::from_json(query_val)?)),
        "boost" => Ok(FtsQuery::Boost(BoostQuery::from_json(query_val)?)),
        "multi_match" => Ok(FtsQuery::MultiMatch(MultiMatchQuery::from_json(query_val)?)),
        "combined_fields" => Ok(FtsQuery::CombinedFields(CombinedFieldsQuery::from_json(
            query_val,
        )?)),
        "boolean" => Ok(FtsQuery::Boolean(BooleanQuery::from_json(query_val)?)),
        _ => Err(Error::invalid_input(format!(
            "unknown fts query type: {}",
            query_type
        ))),
    }
}

pub fn from_json(json: &str) -> Result<FtsQuery> {
    let value: Value = serde_json::from_str(json)
        .map_err(|e| Error::invalid_input(format!("invalid json: {}", e)))?;
    from_json_value(&value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_json_match() {
        let json = r#"
        {
            "match": {
                "column": "text",
                "terms": "hello world",
                "boost": 2.0,
                "fuzziness": 1,
                "max_expansions": 10,
                "operator": "and",
                "prefix_length": 2
            }
        }
        "#;
        let fts_query = from_json(json).unwrap();
        let expected_query = FtsQuery::Match(MatchQuery {
            column: Some("text".to_string()),
            terms: "hello world".to_string(),
            boost: 2.0,
            fuzziness: Some(1),
            max_expansions: 10,
            operator: Operator::And,
            prefix_length: 2,
            document_granularity: None,
        });
        assert_eq!(fts_query, expected_query);
    }

    #[test]
    fn test_from_json_match_invalid_fuzziness_rejected() {
        // Present-but-invalid fuzziness must be rejected instead of silently
        // coercing to exact match (Some(0)) or truncating modulo 2^32.
        for fuzziness in ["4294967296", "\"fuzzy\"", "\"1\"", "1.5", "-1", "true"] {
            let json = format!(
                r#"{{ "match": {{ "column": "text", "terms": "hello", "fuzziness": {fuzziness} }} }}"#
            );
            let error = from_json(&json).unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
            assert!(
                error.to_string().contains("fuzziness"),
                "fuzziness={fuzziness}: {error}"
            );
        }

        // "auto", in any case like the operator, keeps its "determine
        // automatically" meaning.
        for auto in ["auto", "AUTO"] {
            let json = format!(
                r#"{{ "match": {{ "column": "text", "terms": "hello", "fuzziness": "{auto}" }} }}"#
            );
            let FtsQuery::Match(query) = from_json(&json).unwrap() else {
                panic!("expected match query");
            };
            assert_eq!(query.fuzziness, None);
        }
    }

    #[test]
    fn test_from_json_invalid_integer_fields_rejected() {
        for (field, value) in [
            ("max_expansions", "10.0"),
            ("max_expansions", "-1"),
            ("max_expansions", "\"10\""),
            ("prefix_length", "4294967296"),
            ("prefix_length", "2.0"),
            ("slop", "1.0"),
            ("slop", "4294967297"),
        ] {
            let query = if field == "slop" { "phrase" } else { "match" };
            let json = format!(
                r#"{{ "{query}": {{ "column": "text", "terms": "hello", "{field}": {value} }} }}"#
            );
            let error = from_json(&json).unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
            assert!(
                error.to_string().contains(field) && error.to_string().contains(value),
                "{field}={value}: {error}"
            );
        }

        // An explicit null keeps the default.
        let json = r#"{ "match": { "column": "text", "terms": "hello", "max_expansions": null, "prefix_length": null } }"#;
        let FtsQuery::Match(query) = from_json(json).unwrap() else {
            panic!("expected match query");
        };
        assert_eq!(query.max_expansions, MatchQuery::default_max_expansions());
        assert_eq!(query.prefix_length, 0);
    }

    #[test]
    fn test_from_json_match_invalid_operator_rejected() {
        // An operator string other than and/or must error, exactly as the
        // combined_fields parser does, instead of silently running as Or.
        let json =
            r#"{ "match": { "column": "text", "terms": "hello world", "operator": "bogus" } }"#;
        let error = from_json(json).unwrap_err();
        assert!(
            error.to_string().contains("Invalid operator: bogus"),
            "{error}"
        );

        // multi_match sub-queries are parsed by MatchQuery::from_json as well.
        let json = r#"{
            "multi_match": {
                "match_queries": [
                    { "column": "title", "terms": "hello", "operator": "bogus" }
                ]
            }
        }"#;
        let error = from_json(json).unwrap_err();
        assert!(
            error.to_string().contains("Invalid operator: bogus"),
            "{error}"
        );
    }

    #[test]
    fn test_from_json_phrase() {
        let json = r#"
        {
            "phrase": {
                "column": "text",
                "terms": "hello world",
                "slop": 1
            }
        }"#;
        let fts_query = from_json(json).unwrap();
        let expected_query = FtsQuery::Phrase(PhraseQuery {
            column: Some("text".to_string()),
            terms: "hello world".to_string(),
            slop: 1,
            document_granularity: None,
        });
        assert_eq!(fts_query, expected_query);
    }

    #[test]
    fn test_from_json_boost() {
        let json = r#"
        {
            "boost": {
                "positive": {
                    "match": {
                        "column": "title",
                        "terms": "hello"
                    }
                },
                "negative": {
                    "phrase": {
                        "column": "body",
                        "terms": "world"
                    }
                },
                "negative_boost": 0.5
            }
        }"#;
        let fts_query = from_json(json).unwrap();
        let positive_query = Box::new(FtsQuery::Match(
            MatchQuery::new("hello".to_string())
                .with_column(Some("title".to_string()))
                .with_fuzziness(None),
        ));
        let negative_query = Box::new(FtsQuery::Phrase(
            PhraseQuery::new("world".to_string()).with_column(Some("body".to_string())),
        ));
        let expected_query = FtsQuery::Boost(BoostQuery {
            positive: positive_query,
            negative: negative_query,
            negative_boost: 0.5,
        });
        assert_eq!(fts_query, expected_query);
    }

    #[test]
    fn test_from_json_multi_match() {
        let json = r#"
        {
            "multi_match": {
                "match_queries": [
                    {
                        "column": "title",
                        "terms": "hello"
                    },
                    {
                        "column": "body",
                        "terms": "world"
                    }
                ]
            }
        }"#;
        let fts_query = from_json(json).unwrap();
        let match_queries = vec![
            MatchQuery::new("hello".to_string())
                .with_column(Some("title".to_string()))
                .with_fuzziness(None),
            MatchQuery::new("world".to_string())
                .with_column(Some("body".to_string()))
                .with_fuzziness(None),
        ];
        let expected_query = FtsQuery::MultiMatch(MultiMatchQuery { match_queries });
        assert_eq!(fts_query, expected_query);
    }

    #[test]
    fn test_from_json_boolean() {
        let json = r#"{
            "boolean": {
                "must": [
                    {
                        "match": {
                            "column": "text",
                            "terms": "hello"
                        }
                    }
                ],
                "should": [
                    {
                        "phrase": {
                            "column": "text",
                            "terms": "world"
                        }
                    }
                ],
                "must_not": []
            }
        }"#;
        let fts_query = from_json(json).unwrap();
        let must_query = FtsQuery::Match(
            MatchQuery::new("hello".to_string())
                .with_column(Some("text".to_string()))
                .with_fuzziness(None),
        );
        let should_query = FtsQuery::Phrase(
            PhraseQuery::new("world".to_string()).with_column(Some("text".to_string())),
        );

        let expected_query = FtsQuery::Boolean(BooleanQuery::new(vec![
            (Occur::Must, must_query),
            (Occur::Should, should_query),
        ]));
        assert_eq!(fts_query, expected_query);
    }

    #[test]
    fn test_from_json_combined_fields() {
        let json = r#"
        {
            "combined_fields": {
                "query": "hello world",
                "columns": ["title", "body"],
                "boost": [2.0, 1.0],
                "operator": "and"
            }
        }"#;
        let fts_query = from_json(json).unwrap();
        let expected = CombinedFieldsQuery::try_new(
            "hello world".to_string(),
            vec!["title".to_string(), "body".to_string()],
        )
        .unwrap()
        .try_with_boosts(vec![2.0, 1.0])
        .unwrap()
        .with_operator(Operator::And);
        assert_eq!(fts_query, FtsQuery::CombinedFields(expected));
    }

    #[test]
    fn test_from_json_combined_fields_defaults_and_validation() {
        // boost + operator omitted: weights default to 1.0, operator to Or.
        let json = r#"{ "combined_fields": { "query": "hi", "columns": ["a", "b"] } }"#;
        let FtsQuery::CombinedFields(query) = from_json(json).unwrap() else {
            panic!("expected combined_fields query");
        };
        assert_eq!(
            query.weighted_columns().collect::<Vec<_>>(),
            vec![("a", 1.0), ("b", 1.0)]
        );
        assert_eq!(query.operator(), Operator::Or);

        // A weight below 1 is rejected.
        let json = r#"{ "combined_fields": { "query": "hi", "columns": ["a", "b"], "boost": [0.5, 1.0] } }"#;
        assert!(from_json(json).is_err());

        // So is a weight above the upper bound that keeps the blended length
        // finite.
        let json = r#"{ "combined_fields": { "query": "hi", "columns": ["a", "b"], "boost": [1.0, 1e30] } }"#;
        assert!(from_json(json).is_err());

        // Empty columns are rejected.
        let json = r#"{ "combined_fields": { "query": "hi", "columns": [] } }"#;
        assert!(from_json(json).is_err());
    }
}
