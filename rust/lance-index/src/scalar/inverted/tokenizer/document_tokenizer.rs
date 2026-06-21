// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_schema::{DataType, Field};
use lance_arrow::ARROW_EXT_NAME_KEY;
use lance_arrow::json::JSON_EXT_NAME;
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
        match field.data_type() {
            DataType::Utf8 | DataType::LargeUtf8 => Ok(Self::Text),
            DataType::List(field) | DataType::LargeList(field)
                if matches!(field.data_type(), DataType::Utf8 | DataType::LargeUtf8) =>
            {
                Ok(Self::Text)
            }
            DataType::LargeBinary => match field.metadata().get(ARROW_EXT_NAME_KEY) {
                Some(name) if name.as_str() == JSON_EXT_NAME => Ok(Self::Json),
                _ => Err(lance_core::Error::invalid_input_source(
                    format!("field {} is not json", field.name()).into(),
                )),
            },
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
/// `LanceTokenizer` defines 2 methods for tokenization, normally they are the same, but sometimes
/// tokenizer needs different behavior for search and index. Take json document as an example:
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
        while let Some(token) = stream.next() {
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
    /// Whether flattened JSON tokenization avoids cross-array unnesting.
    fn disable_cross_array_unnest(&self) -> bool {
        false
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
    pub fn new(
        tokenizer: TextAnalyzer,
        mode: JsonTokenizerMode,
        disable_cross_array_unnest: bool,
    ) -> Self {
        Self {
            tokenizer,
            mode,
            disable_cross_array_unnest,
        }
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
        let tokens = flatten_triplet(query_text, self.mode, &mut self.tokenizer).unwrap();
        BoxTokenStream::new(TTStream { tokens, index: 0 })
    }

    fn token_stream_for_doc<'a>(&'a mut self, text: &'a str) -> BoxTokenStream<'a> {
        let tokens = self
            .token_streams_for_doc(text)
            .unwrap()
            .into_iter()
            .next()
            .unwrap_or_default();
        BoxTokenStream::new(TTStream { tokens, index: 0 })
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
            JsonTokenizerMode::FlattenedSubDocs => Ok(flatten_json_sub_docs(
                &value,
                "",
                &mut self.tokenizer,
                self.disable_cross_array_unnest,
            )),
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

    fn disable_cross_array_unnest(&self) -> bool {
        self.disable_cross_array_unnest
    }
}

fn flatten_triplet(
    text: &str,
    mode: JsonTokenizerMode,
    tokenizer: &mut TextAnalyzer,
) -> lance_core::Result<Vec<Token>> {
    let mut token_vec = Vec::new();
    let mut idx = 0;

    for triple in text.split(';') {
        let parts: Vec<&str> = triple.splitn(3, ',').collect();
        if parts.len() != 3 {
            return Err(lance_core::Error::invalid_input_source(
                format!("Invalid triple format: {}", triple).into(),
            ));
        }
        let field = parts[0];
        let v_type = parts[1];
        let value = parts[2];
        let (field, mut index_tokens) = match mode {
            JsonTokenizerMode::SingleDocument => (field.to_string(), Vec::new()),
            JsonTokenizerMode::FlattenedSubDocs => normalize_flattened_json_path(field)?,
        };

        for index_token in index_tokens.drain(..) {
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
                return Err(lance_core::Error::invalid_input_source(
                    format!("Invalid triple type: {}", v_type).into(),
                ));
            }
        }
    }
    Ok(token_vec)
}

fn normalize_flattened_json_path(path: &str) -> lance_core::Result<(String, Vec<String>)> {
    let mut normalized = String::with_capacity(path.len());
    let mut index_tokens = Vec::new();
    let mut chars = path.char_indices().peekable();

    while let Some((_, ch)) = chars.next() {
        if ch != '[' {
            normalized.push(ch);
            continue;
        }

        let index_path = normalized.clone();
        let mut array_index = String::new();
        let mut found_right_bracket = false;
        for (_, bracket_ch) in chars.by_ref() {
            if bracket_ch == ']' {
                found_right_bracket = true;
                break;
            }
            array_index.push(bracket_ch);
        }
        if !found_right_bracket {
            return Err(Error::invalid_input(format!(
                "missing right bracket in JSON path {path:?}"
            )));
        }
        if array_index.is_empty() {
            return Err(Error::invalid_input(format!(
                "empty array index in JSON path {path:?}"
            )));
        }
        if array_index != "*" {
            index_tokens.push(format!("{index_path}$idx,number,{array_index}"));
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

fn flatten_json_sub_docs(
    value: &Value,
    prefix: &str,
    tokenizer: &mut TextAnalyzer,
    disable_cross_array_unnest: bool,
) -> Vec<Vec<Token>> {
    let token_texts =
        flatten_json_sub_doc_terms(value, prefix, tokenizer, disable_cross_array_unnest);
    token_texts
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

fn flatten_json_sub_doc_terms(
    value: &Value,
    prefix: &str,
    tokenizer: &mut TextAnalyzer,
    disable_cross_array_unnest: bool,
) -> Vec<Vec<String>> {
    match value {
        Value::Object(map) => {
            let mut non_nested = Vec::new();
            let mut nested: Vec<Vec<Vec<String>>> = Vec::new();

            for (key, child) in map {
                let child_prefix = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                let child_terms = flatten_json_sub_doc_terms(
                    child,
                    &child_prefix,
                    tokenizer,
                    disable_cross_array_unnest,
                );
                match child_terms.len() {
                    0 => {}
                    1 => non_nested.extend(child_terms.into_iter().next().unwrap()),
                    _ => nested.push(child_terms),
                }
            }

            match nested.len() {
                0 if non_nested.is_empty() => Vec::new(),
                0 => vec![non_nested],
                1 => nested
                    .pop()
                    .unwrap()
                    .into_iter()
                    .map(|mut sub_doc| {
                        sub_doc.extend(non_nested.iter().cloned());
                        sub_doc
                    })
                    .collect(),
                _ if disable_cross_array_unnest => unnest_json_sub_docs(&nested, &non_nested),
                _ => cross_join_json_sub_docs(&nested, &non_nested),
            }
        }
        Value::Array(arr) => {
            let mut sub_docs = Vec::new();
            let child_prefix = format!("{prefix}.");
            for (array_index, child) in arr.iter().enumerate() {
                let mut child_terms = flatten_json_sub_doc_terms(
                    child,
                    &child_prefix,
                    tokenizer,
                    disable_cross_array_unnest,
                );
                for sub_doc in &mut child_terms {
                    sub_doc.push(format!("{prefix}$idx,number,{array_index}"));
                }
                sub_docs.extend(child_terms);
            }
            sub_docs
        }
        Value::String(text) => {
            let mut token_texts = Vec::new();
            let mut tokens = tokenizer.token_stream(text);
            while let Some(token) = tokens.next() {
                token_texts.push(format!("{prefix},str,{}", token.text));
            }
            if token_texts.is_empty() {
                Vec::new()
            } else {
                vec![token_texts]
            }
        }
        _ => {
            let value_type = match value {
                Value::Null => "null",
                Value::Bool(_) => "bool",
                Value::Number(_) => "number",
                _ => unreachable!(),
            };
            vec![vec![format!("{prefix},{value_type},{value}")]]
        }
    }
}

fn cross_join_json_sub_docs(
    nested: &[Vec<Vec<String>>],
    non_nested: &[String],
) -> Vec<Vec<String>> {
    let capacity = nested
        .iter()
        .map(|sub_docs| sub_docs.len())
        .product::<usize>();
    let mut results = Vec::with_capacity(capacity);
    let mut current = Vec::new();
    cross_join_json_sub_docs_inner(nested, 0, non_nested, &mut current, &mut results);
    results
}

fn unnest_json_sub_docs(nested: &[Vec<Vec<String>>], non_nested: &[String]) -> Vec<Vec<String>> {
    let capacity = nested.iter().map(|sub_docs| sub_docs.len()).sum::<usize>();
    let mut results = Vec::with_capacity(capacity);
    for sub_docs in nested {
        for child in sub_docs {
            let mut sub_doc = child.clone();
            sub_doc.extend(non_nested.iter().cloned());
            results.push(sub_doc);
        }
    }
    results
}

fn cross_join_json_sub_docs_inner(
    nested: &[Vec<Vec<String>>],
    nested_index: usize,
    non_nested: &[String],
    current: &mut Vec<String>,
    results: &mut Vec<Vec<String>>,
) {
    if nested_index == nested.len() {
        let mut sub_doc = current.clone();
        sub_doc.extend(non_nested.iter().cloned());
        results.push(sub_doc);
        return;
    }

    for child in &nested[nested_index] {
        let old_len = current.len();
        current.extend(child.iter().cloned());
        cross_join_json_sub_docs_inner(nested, nested_index + 1, non_nested, current, results);
        current.truncate(old_len);
    }
}

struct TTStream {
    tokens: Vec<Token>,
    index: usize,
}

impl TokenStream for TTStream {
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
    use crate::scalar::inverted::tokenizer::document_tokenizer::{
        JsonTokenizer, JsonTokenizerMode, LanceTokenizer, flatten_json, flatten_json_sub_docs,
        flatten_triplet,
    };
    use lance_tokenizer::{SimpleTokenizer, TextAnalyzer, Token};
    use serde_json::Value;

    #[test]
    fn test_json_tokenizer() {
        let text = r#"{
          "a": 1,
          "b": [
            {"c": "d"},
            {"c": "e"}
          ]
        }"#;
        let mut tokenizer = JsonTokenizer::new(
            TextAnalyzer::builder(SimpleTokenizer::default()).build(),
            JsonTokenizerMode::SingleDocument,
            false,
        );
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

    #[test]
    fn test_flattened_sub_docs_design_example() {
        let doc0 = flattened_sub_doc_texts(r#"{"foo":[{"bar":["x","y"]}]}"#);
        let doc1 = flattened_sub_doc_texts(r#"{"foo":[{"bar":["y"]},{"bar":"z"}]}"#);

        assert_eq!(
            doc0,
            vec![
                sorted_tokens([
                    "foo$idx,number,0",
                    "foo..bar$idx,number,0",
                    "foo..bar.,str,x",
                ]),
                sorted_tokens([
                    "foo$idx,number,0",
                    "foo..bar$idx,number,1",
                    "foo..bar.,str,y",
                ]),
            ]
        );
        assert_eq!(
            doc1,
            vec![
                sorted_tokens([
                    "foo$idx,number,0",
                    "foo..bar$idx,number,0",
                    "foo..bar.,str,y",
                ]),
                sorted_tokens(["foo$idx,number,1", "foo..bar,str,z"]),
            ]
        );

        let mut tokenizer = TextAnalyzer::builder(SimpleTokenizer::default()).build();
        let exact_tokens = flatten_triplet(
            "foo[0].bar[0],str,y",
            JsonTokenizerMode::FlattenedSubDocs,
            &mut tokenizer,
        )
        .unwrap();
        assert_token_texts(
            &exact_tokens,
            &[
                "foo$idx,number,0",
                "foo..bar$idx,number,0",
                "foo..bar.,str,y",
            ],
        );

        let wildcard_tokens = flatten_triplet(
            "foo[0].bar[*],str,y",
            JsonTokenizerMode::FlattenedSubDocs,
            &mut tokenizer,
        )
        .unwrap();
        assert_token_texts(&wildcard_tokens, &["foo$idx,number,0", "foo..bar.,str,y"]);
    }

    #[test]
    fn test_flattened_sub_docs_sibling_array_example() {
        let doc0 = flattened_sub_doc_texts(
            r#"{"foo":[{"bar":["x","y"]},{"bar":["a","b"]}],"foo2":["u"]}"#,
        );
        let doc1 = flattened_sub_doc_texts(r#"{"foo":[{"bar":["y","z"]}],"foo2":["u"]}"#);

        assert_eq!(
            doc0,
            vec![
                sorted_tokens([
                    "foo$idx,number,0",
                    "foo..bar$idx,number,0",
                    "foo..bar.,str,x",
                    "foo2$idx,number,0",
                    "foo2.,str,u",
                ]),
                sorted_tokens([
                    "foo$idx,number,0",
                    "foo..bar$idx,number,1",
                    "foo..bar.,str,y",
                    "foo2$idx,number,0",
                    "foo2.,str,u",
                ]),
                sorted_tokens([
                    "foo$idx,number,1",
                    "foo..bar$idx,number,0",
                    "foo..bar.,str,a",
                    "foo2$idx,number,0",
                    "foo2.,str,u",
                ]),
                sorted_tokens([
                    "foo$idx,number,1",
                    "foo..bar$idx,number,1",
                    "foo..bar.,str,b",
                    "foo2$idx,number,0",
                    "foo2.,str,u",
                ]),
            ]
        );
        assert_eq!(
            doc1,
            vec![
                sorted_tokens([
                    "foo$idx,number,0",
                    "foo..bar$idx,number,0",
                    "foo..bar.,str,y",
                    "foo2$idx,number,0",
                    "foo2.,str,u",
                ]),
                sorted_tokens([
                    "foo$idx,number,0",
                    "foo..bar$idx,number,1",
                    "foo..bar.,str,z",
                    "foo2$idx,number,0",
                    "foo2.,str,u",
                ]),
            ]
        );
    }

    #[test]
    fn test_disable_cross_array_unnest_indexes_arrays_independently() {
        let cross_joined = flattened_sub_doc_texts(r#"{"a":["x","y"],"b":["u","v"],"c":1}"#);
        let disabled = flattened_sub_doc_texts_with_disable_cross_array_unnest(
            r#"{"a":["x","y"],"b":["u","v"],"c":1}"#,
        );

        assert_eq!(
            sorted_sub_docs(cross_joined),
            sorted_sub_docs(vec![
                sorted_tokens([
                    "a$idx,number,0",
                    "a.,str,x",
                    "b$idx,number,0",
                    "b.,str,u",
                    "c,number,1"
                ]),
                sorted_tokens([
                    "a$idx,number,0",
                    "a.,str,x",
                    "b$idx,number,1",
                    "b.,str,v",
                    "c,number,1"
                ]),
                sorted_tokens([
                    "a$idx,number,1",
                    "a.,str,y",
                    "b$idx,number,0",
                    "b.,str,u",
                    "c,number,1"
                ]),
                sorted_tokens([
                    "a$idx,number,1",
                    "a.,str,y",
                    "b$idx,number,1",
                    "b.,str,v",
                    "c,number,1"
                ]),
            ])
        );
        assert_eq!(
            sorted_sub_docs(disabled),
            sorted_sub_docs(vec![
                sorted_tokens(["a$idx,number,0", "a.,str,x", "c,number,1"]),
                sorted_tokens(["a$idx,number,1", "a.,str,y", "c,number,1"]),
                sorted_tokens(["b$idx,number,0", "b.,str,u", "c,number,1"]),
                sorted_tokens(["b$idx,number,1", "b.,str,v", "c,number,1"]),
            ])
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

    fn flattened_sub_doc_texts(json: &str) -> Vec<Vec<String>> {
        flattened_sub_doc_texts_with_mode(json, false)
    }

    fn flattened_sub_doc_texts_with_disable_cross_array_unnest(json: &str) -> Vec<Vec<String>> {
        flattened_sub_doc_texts_with_mode(json, true)
    }

    fn flattened_sub_doc_texts_with_mode(
        json: &str,
        disable_cross_array_unnest: bool,
    ) -> Vec<Vec<String>> {
        let value: Value = serde_json::from_str(json).unwrap();
        let mut tokenizer = TextAnalyzer::builder(SimpleTokenizer::default()).build();
        flatten_json_sub_docs(&value, "", &mut tokenizer, disable_cross_array_unnest)
            .into_iter()
            .map(|tokens| sorted_tokens(tokens.into_iter().map(|token| token.text)))
            .collect()
    }

    fn sorted_tokens(tokens: impl IntoIterator<Item = impl Into<String>>) -> Vec<String> {
        let mut tokens = tokens.into_iter().map(Into::into).collect::<Vec<String>>();
        tokens.sort();
        tokens
    }

    fn sorted_sub_docs(mut sub_docs: Vec<Vec<String>>) -> Vec<Vec<String>> {
        sub_docs.sort();
        sub_docs
    }

    fn assert_token_texts(tokens: &[Token], expected: &[&str]) {
        let actual = tokens
            .iter()
            .map(|token| token.text.as_str())
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }
}
