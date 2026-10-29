// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_schema::{DataType, Field};
use lance_arrow::json::JsonEncoding;
use lance_core::Error;
use lance_tokenizer::{BoxTokenStream, TextAnalyzer, Token, TokenStream};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::str::FromStr;

/// Document type for full text search.
#[derive(Debug, Clone)]
pub enum DocType {
    Text,
    Json,
}

/// Controls how JSON documents are represented inside the inverted index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JsonTokenizerMode {
    /// Emit one token stream for each source JSON document.
    SingleDocument,
    /// Flatten arrays into multiple sub-doc token streams for each source JSON document.
    FlattenedSubDocs,
}

impl AsRef<str> for JsonTokenizerMode {
    fn as_ref(&self) -> &str {
        match self {
            Self::SingleDocument => "single_document",
            Self::FlattenedSubDocs => "flattened_sub_docs",
        }
    }
}

impl FromStr for JsonTokenizerMode {
    type Err = Error;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "single_document" => Ok(Self::SingleDocument),
            "flattened_sub_docs" => Ok(Self::FlattenedSubDocs),
            _ => Err(Error::invalid_input(format!(
                "unknown JSON tokenizer mode {value:?}; expected 'single_document' or 'flattened_sub_docs'"
            ))),
        }
    }
}

impl AsRef<str> for DocType {
    fn as_ref(&self) -> &str {
        match self {
            Self::Text => "text",
            Self::Json => "json",
        }
    }
}

impl TryFrom<&Field> for DocType {
    type Error = lance_core::Error;

    fn try_from(field: &Field) -> Result<Self, Self::Error> {
        // JSON text is also `Utf8`, so it must be recognized before plain text.
        if JsonEncoding::of_field(field).is_some() {
            return Ok(Self::Json);
        }
        match field.data_type() {
            DataType::Utf8 | DataType::LargeUtf8 => Ok(Self::Text),
            DataType::List(field) | DataType::LargeList(field)
                if matches!(field.data_type(), DataType::Utf8 | DataType::LargeUtf8) =>
            {
                Ok(Self::Text)
            }
            _ => Err(lance_core::Error::invalid_input_source(
                format!("field {} is not json", field.name()).into(),
            )),
        }
    }
}

impl DocType {
    /// Get the length of the prefix before value.
    ///  - JSON Token: path,type,value
    ///  - Text Token: value
    pub fn prefix_len(&self, token: &str) -> usize {
        match self {
            Self::Json => {
                if let Some(pos) = token.find(',')
                    && let Some(second_pos) = token[pos + 1..].find(',')
                {
                    return pos + second_pos + 2;
                }
                panic!("json token must be in format of <path>,<type>,<value>")
            }
            Self::Text => 0,
        }
    }
}

/// Lance full text search tokenizer.
///
/// Search text and indexed documents can require different tokenization. For JSON:
/// 1. Query text is a triplet <path,type,value>, something like `a.b,str,123`. We shouldn't use
///    json in search, because it would be too complicated.
/// 2. Document text is a json string.
pub trait LanceTokenizer: Send + Sync + std::fmt::Debug {
    /// Tokenize query text for search.
    fn token_stream_for_search<'a>(&'a mut self, query_text: &'a str) -> BoxTokenStream<'a>;
    /// Tokenize document text for index.
    fn token_stream_for_doc<'a>(&'a mut self, text: &'a str) -> BoxTokenStream<'a>;
    /// Tokenize document text into one or more internal inverted-index documents.
    fn token_streams_for_doc(&mut self, text: &str) -> lance_core::Result<Vec<Vec<Token>>> {
        let mut stream = self.token_stream_for_doc(text);
        let mut tokens = Vec::new();
        while let Some(token) = stream.try_next().map_err(Error::invalid_input)? {
            tokens.push(token.clone());
        }
        Ok(vec![tokens])
    }
    /// Clone the tokenizer.
    fn box_clone(&self) -> Box<dyn LanceTokenizer>;
    /// Get document type.
    fn doc_type(&self) -> DocType;
    /// Get the JSON tokenization mode, if this tokenizer handles JSON documents.
    fn json_tokenizer_mode(&self) -> Option<JsonTokenizerMode> {
        None
    }
}

impl Clone for Box<dyn LanceTokenizer> {
    fn clone(&self) -> Self {
        self.box_clone()
    }
}

#[derive(Clone)]
pub struct TextTokenizer {
    tokenizer: TextAnalyzer,
}

impl std::fmt::Debug for TextTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TextTokenizer")
    }
}

impl TextTokenizer {
    pub fn new(tokenizer: TextAnalyzer) -> Self {
        Self { tokenizer }
    }
}

impl LanceTokenizer for TextTokenizer {
    fn token_stream_for_search<'a>(&'a mut self, query_text: &'a str) -> BoxTokenStream<'a> {
        self.tokenizer.token_stream(query_text)
    }

    fn token_stream_for_doc<'a>(&'a mut self, text: &'a str) -> BoxTokenStream<'a> {
        self.tokenizer.token_stream(text)
    }

    fn box_clone(&self) -> Box<dyn LanceTokenizer> {
        Box::new(self.clone())
    }

    fn doc_type(&self) -> DocType {
        DocType::Text
    }
}

#[derive(Clone)]
pub struct JsonTokenizer {
    tokenizer: TextAnalyzer,
    mode: JsonTokenizerMode,
    disable_cross_array_unnest: bool,
}

impl JsonTokenizer {
    pub fn new(tokenizer: TextAnalyzer) -> Self {
        Self {
            tokenizer,
            mode: JsonTokenizerMode::SingleDocument,
            disable_cross_array_unnest: false,
        }
    }

    #[doc(hidden)]
    pub fn with_mode(mut self, mode: JsonTokenizerMode) -> Self {
        self.mode = mode;
        self
    }

    pub(crate) fn with_disable_cross_array_unnest(
        mut self,
        disable_cross_array_unnest: bool,
    ) -> Self {
        self.disable_cross_array_unnest = disable_cross_array_unnest;
        self
    }
}

impl std::fmt::Debug for JsonTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsonTokenizer")
            .field("mode", &self.mode)
            .field(
                "disable_cross_array_unnest",
                &self.disable_cross_array_unnest,
            )
            .finish()
    }
}

impl LanceTokenizer for JsonTokenizer {
    fn token_stream_for_search<'a>(&'a mut self, query_text: &'a str) -> BoxTokenStream<'a> {
        match flatten_triplet(query_text, self.mode, &mut self.tokenizer) {
            Ok(tokens) => BoxTokenStream::new(OwnedTokenStream::new(tokens)),
            Err(error) => BoxTokenStream::new(OwnedTokenStream::new(Vec::new())).with_error(error),
        }
    }

    fn token_stream_for_doc<'a>(&'a mut self, text: &'a str) -> BoxTokenStream<'a> {
        match self.token_streams_for_doc(text) {
            Ok(documents) => BoxTokenStream::new(OwnedTokenStream::new(
                documents.into_iter().next().unwrap_or_default(),
            )),
            Err(error) => {
                BoxTokenStream::new(OwnedTokenStream::new(Vec::new())).with_error(error.to_string())
            }
        }
    }

    fn token_streams_for_doc(&mut self, text: &str) -> lance_core::Result<Vec<Vec<Token>>> {
        let value: Value = serde_json::from_slice(text.as_bytes()).map_err(|err| {
            Error::invalid_input(format!(
                "failed to parse JSON document for FTS indexing: {err}"
            ))
        })?;

        match self.mode {
            JsonTokenizerMode::SingleDocument => {
                let mut tokens = Vec::new();
                let mut position = 0;
                flatten_json(&value, "", &mut tokens, &mut position, &mut self.tokenizer);
                Ok(vec![tokens])
            }
            JsonTokenizerMode::FlattenedSubDocs => Ok(JsonFlattener {
                tokenizer: &mut self.tokenizer,
                disable_cross_array_unnest: self.disable_cross_array_unnest,
            }
            .tokenize(&value)),
        }
    }

    fn box_clone(&self) -> Box<dyn LanceTokenizer> {
        Box::new(self.clone())
    }

    fn doc_type(&self) -> DocType {
        DocType::Json
    }

    fn json_tokenizer_mode(&self) -> Option<JsonTokenizerMode> {
        Some(self.mode)
    }
}

fn flatten_triplet(
    text: &str,
    mode: JsonTokenizerMode,
    tokenizer: &mut TextAnalyzer,
) -> Result<Vec<Token>, String> {
    let mut token_vec = Vec::new();
    let mut idx = 0;

    for triple in text.split(';') {
        let parts: Vec<&str> = triple.splitn(3, ',').collect();
        if parts.len() != 3 {
            return Err(format!("Invalid triple format: {}", triple));
        }
        let field = parts[0];
        let v_type = parts[1];
        let value = parts[2];
        let (field, index_tokens) = match mode {
            JsonTokenizerMode::SingleDocument => (field.to_string(), Vec::new()),
            JsonTokenizerMode::FlattenedSubDocs => normalize_flattened_json_path(field)?,
        };

        for index_token in index_tokens {
            token_vec.push(Token {
                offset_from: 0,
                offset_to: 0,
                position: idx,
                text: index_token,
                position_length: 1,
            });
            idx += 1;
        }

        match v_type {
            "number" | "bool" | "null" => {
                let token = Token {
                    offset_from: 0,
                    offset_to: 0,
                    position: idx,
                    text: format!("{},{},{}", field, v_type, value),
                    position_length: 1,
                };
                token_vec.push(token);
                idx += 1;
            }
            "str" => {
                let mut tokens = tokenizer.token_stream(value);
                while let Some(token) = tokens.next() {
                    token_vec.push(Token {
                        offset_from: 0,
                        offset_to: 0,
                        position: idx,
                        text: format!("{},{},{}", field, v_type, token.text),
                        position_length: 1,
                    });
                    idx += 1;
                }
            }
            _ => {
                return Err(format!("Invalid triple type: {}", v_type));
            }
        }
    }
    Ok(token_vec)
}

fn normalize_flattened_json_path(path: &str) -> Result<(String, Vec<String>), String> {
    let mut normalized = String::with_capacity(path.len());
    let mut index_tokens = Vec::new();
    let mut chars = path.chars();

    while let Some(ch) = chars.next() {
        if ch != '[' {
            normalized.push(ch);
            continue;
        }

        let mut array_index = String::new();
        let mut found_right_bracket = false;
        for bracket_ch in chars.by_ref() {
            if bracket_ch == ']' {
                found_right_bracket = true;
                break;
            }
            array_index.push(bracket_ch);
        }
        if !found_right_bracket {
            return Err(format!("missing right bracket in JSON path {path:?}"));
        }
        if array_index.is_empty() {
            return Err(format!("empty array index in JSON path {path:?}"));
        }
        if array_index != "*" {
            index_tokens.push(format!("{normalized}$idx,number,{array_index}"));
        }
        normalized.push('.');
    }

    Ok((normalized, index_tokens))
}

fn flatten_json(
    value: &Value,
    prefix: &str,
    out: &mut Vec<Token>,
    position: &mut usize,
    tokenizer: &mut TextAnalyzer,
) {
    match value {
        Value::Object(map) => {
            for (k, v) in map {
                let next_prefix = if prefix.is_empty() {
                    k.clone()
                } else {
                    format!("{}.{}", prefix, k)
                };
                flatten_json(v, &next_prefix, out, position, tokenizer);
            }
        }
        Value::Array(arr) => {
            for v in arr.iter() {
                flatten_json(v, prefix, out, position, tokenizer);
            }
        }
        Value::String(text) => {
            let mut tokens = tokenizer.token_stream(text);
            while let Some(token) = tokens.next() {
                let token = Token {
                    offset_from: 0,
                    offset_to: 0,
                    position: *position,
                    text: format!("{},{},{}", prefix, "str", token.text),
                    position_length: 1,
                };
                *position += 1;
                out.push(token);
            }
        }
        _ => {
            let value_type = match value {
                Value::Null => "null",
                Value::Bool(_) => "bool",
                Value::Number(_) => "number",
                _ => unreachable!(),
            };
            let token = Token {
                offset_from: 0,
                offset_to: 0,
                position: *position,
                text: format!("{},{},{}", prefix, value_type, value),
                position_length: 1,
            };
            *position += 1;
            out.push(token);
        }
    }
}

struct FlattenedJsonSubDocs {
    sub_docs: Vec<Vec<String>>,
    has_array: bool,
}

struct JsonFlattener<'a> {
    tokenizer: &'a mut TextAnalyzer,
    disable_cross_array_unnest: bool,
}

impl JsonFlattener<'_> {
    fn tokenize(&mut self, value: &Value) -> Vec<Vec<Token>> {
        self.flatten(value, "")
            .sub_docs
            .into_iter()
            .map(|sub_doc| {
                sub_doc
                    .into_iter()
                    .enumerate()
                    .map(|(position, text)| Token {
                        offset_from: 0,
                        offset_to: 0,
                        position,
                        text,
                        position_length: 1,
                    })
                    .collect()
            })
            .collect()
    }

    fn flatten(&mut self, value: &Value, prefix: &str) -> FlattenedJsonSubDocs {
        match value {
            Value::Object(map) => {
                let mut scalar_terms = Vec::new();
                let mut array_groups = Vec::new();
                for (key, child) in map {
                    let child_prefix = if prefix.is_empty() {
                        key.clone()
                    } else {
                        format!("{prefix}.{key}")
                    };
                    let child = self.flatten(child, &child_prefix);
                    if child.has_array {
                        if !child.sub_docs.is_empty() {
                            array_groups.push(child.sub_docs);
                        }
                    } else {
                        scalar_terms.extend(child.sub_docs.into_iter().flatten());
                    }
                }
                if array_groups.is_empty() {
                    return FlattenedJsonSubDocs {
                        sub_docs: if scalar_terms.is_empty() {
                            Vec::new()
                        } else {
                            vec![scalar_terms]
                        },
                        has_array: false,
                    };
                }
                let mut sub_docs = Vec::new();
                if self.disable_cross_array_unnest || array_groups.len() == 1 {
                    sub_docs.extend(array_groups.into_iter().flatten());
                } else {
                    cross_join_json_sub_docs(&array_groups, &mut Vec::new(), &mut sub_docs);
                }
                for sub_doc in &mut sub_docs {
                    sub_doc.extend(scalar_terms.iter().cloned());
                }
                FlattenedJsonSubDocs {
                    sub_docs,
                    has_array: true,
                }
            }
            Value::Array(values) => {
                let mut sub_docs = Vec::new();
                let child_prefix = format!("{prefix}.");
                for (array_index, child) in values.iter().enumerate() {
                    let mut child = self.flatten(child, &child_prefix);
                    for sub_doc in &mut child.sub_docs {
                        sub_doc.push(format!("{prefix}$idx,number,{array_index}"));
                    }
                    sub_docs.extend(child.sub_docs);
                }
                FlattenedJsonSubDocs {
                    sub_docs,
                    has_array: true,
                }
            }
            _ => {
                let terms = match value {
                    Value::String(text) => {
                        let mut terms = Vec::new();
                        let mut stream = self.tokenizer.token_stream(text);
                        while let Some(token) = stream.next() {
                            terms.push(format!("{prefix},str,{}", token.text));
                        }
                        terms
                    }
                    Value::Null => vec![format!("{prefix},null,null")],
                    Value::Bool(value) => vec![format!("{prefix},bool,{value}")],
                    Value::Number(value) => vec![format!("{prefix},number,{value}")],
                    _ => unreachable!(),
                };
                FlattenedJsonSubDocs {
                    sub_docs: if terms.is_empty() {
                        Vec::new()
                    } else {
                        vec![terms]
                    },
                    has_array: false,
                }
            }
        }
    }
}

fn cross_join_json_sub_docs(
    groups: &[Vec<Vec<String>>],
    current: &mut Vec<String>,
    results: &mut Vec<Vec<String>>,
) {
    let Some((group, remaining)) = groups.split_first() else {
        results.push(current.clone());
        return;
    };
    for child in group {
        let old_len = current.len();
        current.extend(child.iter().cloned());
        cross_join_json_sub_docs(remaining, current, results);
        current.truncate(old_len);
    }
}

struct OwnedTokenStream {
    tokens: Vec<Token>,
    index: usize,
}

impl OwnedTokenStream {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, index: 0 }
    }
}

impl TokenStream for OwnedTokenStream {
    fn advance(&mut self) -> bool {
        if self.index < self.tokens.len() {
            self.index += 1;
            true
        } else {
            false
        }
    }

    fn token(&self) -> &Token {
        &self.tokens[self.index - 1]
    }

    fn token_mut(&mut self) -> &mut Token {
        &mut self.tokens[self.index - 1]
    }
}

#[cfg(test)]
mod tests {
    use crate::scalar::inverted::query::try_collect_query_tokens;
    use crate::scalar::inverted::tokenizer::document_tokenizer::{
        DocType, JsonTokenizer, JsonTokenizerMode, LanceTokenizer, flatten_json, flatten_triplet,
    };
    use arrow_schema::{DataType, Field};
    use lance_arrow::ARROW_EXT_NAME_KEY;
    use lance_arrow::json::{ARROW_JSON_EXT_NAME, json_field};
    use lance_core::Error;
    use lance_tokenizer::{SimpleTokenizer, TextAnalyzer, Token};
    use rstest::rstest;
    use serde_json::Value;
    use std::collections::HashMap;

    /// A JSON column is tokenized as JSON whether it holds stored JSONB or
    /// Arrow JSON text; plain strings stay text.
    #[rstest]
    #[case::jsonb(json_field("doc", true), "json")]
    #[case::arrow_json_text(
        Field::new("doc", DataType::Utf8, true).with_metadata(HashMap::from([(
            ARROW_EXT_NAME_KEY.to_string(),
            ARROW_JSON_EXT_NAME.to_string(),
        )])),
        "json"
    )]
    #[case::plain_text(Field::new("doc", DataType::Utf8, true), "text")]
    fn test_doc_type_of_field(#[case] field: Field, #[case] expected: &str) {
        assert_eq!(DocType::try_from(&field).unwrap().as_ref(), expected);
    }

    #[test]
    fn test_json_tokenizer() {
        let text = r#"{
          "a": 1,
          "b": [
            {"c": "d"},
            {"c": "e"}
          ]
        }"#;
        let mut tokenizer =
            JsonTokenizer::new(TextAnalyzer::builder(SimpleTokenizer::default()).build());
        let mut stream = tokenizer.token_stream_for_doc(text);

        let mut tokens: Vec<Token> = vec![];
        while let Some(token) = stream.next() {
            tokens.push(token.clone());
        }

        assert_eq!(tokens.len(), 3);
        assert_token(&tokens[0], 0, "a,number,1");
        assert_token(&tokens[1], 1, "b.c,str,d");
        assert_token(&tokens[2], 2, "b.c,str,e");
    }

    #[test]
    fn test_flatten_json_text() {
        let json = r#"{
              "a": 1,
              "b": [
                {"c": "hello world"},
                {"c": "e"}
              ],
              "c": true,
              "d": null,
              "e": {
                "f": 1.0
              }
          }"#;
        let value: Value = serde_json::from_str(json).unwrap();

        let mut tokens = vec![];
        let mut tokenizer = TextAnalyzer::builder(SimpleTokenizer::default()).build();
        let mut position = 0;
        flatten_json(&value, "", &mut tokens, &mut position, &mut tokenizer);

        assert_eq!(7, tokens.len());
        assert_token(&tokens[0], 0, "a,number,1");
        assert_token(&tokens[1], 1, "b.c,str,hello");
        assert_token(&tokens[2], 2, "b.c,str,world");
        assert_token(&tokens[3], 3, "b.c,str,e");
        assert_token(&tokens[4], 4, "c,bool,true");
        assert_token(&tokens[5], 5, "d,null,null");
        assert_token(&tokens[6], 6, "e.f,number,1.0");
    }

    #[test]
    fn test_flatten_triplet() {
        let text = r#"a,number,1;b.c,str,d;b.c,str,e;d,str,hello world;e,number,1.0"#;
        let mut tokenizer = TextAnalyzer::builder(SimpleTokenizer::default()).build();
        let tokens =
            flatten_triplet(text, JsonTokenizerMode::SingleDocument, &mut tokenizer).unwrap();

        assert_eq!(tokens.len(), 6);
        assert_token(&tokens[0], 0, "a,number,1");
        assert_token(&tokens[1], 1, "b.c,str,d");
        assert_token(&tokens[2], 2, "b.c,str,e");
        assert_token(&tokens[3], 3, "d,str,hello");
        assert_token(&tokens[4], 4, "d,str,world");
        assert_token(&tokens[5], 5, "e,number,1.0");
    }

    #[rstest]
    #[case::missing_type("brown", "Invalid triple format: brown")]
    #[case::invalid_type("title,string,brown", "Invalid triple type: string")]
    fn test_invalid_json_search_query(#[case] query: &str, #[case] expected_message: &str) {
        let mut tokenizer: Box<dyn LanceTokenizer> = Box::new(JsonTokenizer::new(
            TextAnalyzer::builder(SimpleTokenizer::default()).build(),
        ));
        let error = try_collect_query_tokens(query, &mut tokenizer)
            .err()
            .expect("invalid JSON search query should fail");

        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains(expected_message), "{error}");
    }

    #[test]
    fn test_flattened_json_preserves_nested_array_paths() {
        assert_sub_docs(
            r#"{"foo":[{"bar":["x","y"]},{"bar":"z"}]}"#,
            false,
            &[
                "foo$idx,number,0;foo..bar$idx,number,0;foo..bar.,str,x",
                "foo$idx,number,0;foo..bar$idx,number,1;foo..bar.,str,y",
                "foo$idx,number,1;foo..bar,str,z",
            ],
        );
    }

    #[rstest]
    #[case::cartesian_product(
        false,
        &[
            "a$idx,number,0;a.,str,x;b$idx,number,0;b.,str,u;c,number,1",
            "a$idx,number,0;a.,str,x;b$idx,number,1;b.,str,v;c,number,1",
            "a$idx,number,1;a.,str,y;b$idx,number,0;b.,str,u;c,number,1",
            "a$idx,number,1;a.,str,y;b$idx,number,1;b.,str,v;c,number,1",
        ],
    )]
    #[case::independent_siblings(
        true,
        &[
            "a$idx,number,0;a.,str,x;c,number,1",
            "a$idx,number,1;a.,str,y;c,number,1",
            "b$idx,number,0;b.,str,u;c,number,1",
            "b$idx,number,1;b.,str,v;c,number,1",
        ],
    )]
    fn test_sibling_array_expansion(
        #[case] disable_cross_array_unnest: bool,
        #[case] expected_sub_docs: &[&str],
    ) {
        assert_sub_docs(
            r#"{"a":["x","y"],"b":["u","v"],"c":1}"#,
            disable_cross_array_unnest,
            expected_sub_docs,
        );
    }

    fn assert_token(token: &Token, position: usize, text: &str) {
        assert_eq!(
            token.position, position,
            "expected position {position} but {token:?}"
        );
        assert_eq!(
            token.text.as_str(),
            text,
            "expected text {text} but {token:?}"
        );
    }

    fn assert_sub_docs(json: &str, disable_cross_array_unnest: bool, expected: &[&str]) {
        let mut actual =
            JsonTokenizer::new(TextAnalyzer::builder(SimpleTokenizer::default()).build())
                .with_mode(JsonTokenizerMode::FlattenedSubDocs)
                .with_disable_cross_array_unnest(disable_cross_array_unnest)
                .token_streams_for_doc(json)
                .unwrap()
                .into_iter()
                .map(|tokens| sorted_tokens(tokens.into_iter().map(|token| token.text)))
                .collect::<Vec<_>>();
        actual.sort();
        let mut expected = expected
            .iter()
            .map(|sub_doc| sorted_tokens(sub_doc.split(';')))
            .collect::<Vec<_>>();
        expected.sort();
        assert_eq!(actual, expected);
    }

    fn sorted_tokens(tokens: impl IntoIterator<Item = impl Into<String>>) -> Vec<String> {
        let mut tokens = tokens.into_iter().map(Into::into).collect::<Vec<String>>();
        tokens.sort();
        tokens
    }
}
