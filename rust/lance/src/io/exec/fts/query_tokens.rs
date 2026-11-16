// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Display snapshots and tokenization for FTS query trees.
//!
//! Token snapshots are diagnostic state attached to execution nodes. They do
//! not participate in matching or scoring.

use std::collections::HashMap;
use std::sync::OnceLock;

use lance_core::{Error, Result};
use lance_index::scalar::inverted::InvertedIndex;
use lance_index::scalar::inverted::document_tokenizer::{DocType, JsonTokenizer, LanceTokenizer};
use lance_index::scalar::inverted::query::{FtsQuery, Tokens, try_collect_query_tokens};
use lance_index::scalar::inverted::tokenizer::document_tokenizer::TextTokenizer;
use lance_tokenizer::{SimpleTokenizer, TextAnalyzer};

use super::matches::count_fts_leaves;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct TokenWithPosition {
    text: String,
    position: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct TokenizedQuery(Vec<TokenWithPosition>);

impl TokenizedQuery {
    fn from_tokens(tokens: &Tokens) -> Self {
        let mut token_positions = Vec::with_capacity(tokens.len());
        for index in 0..tokens.len() {
            token_positions.push(TokenWithPosition {
                text: tokens.get_token(index).to_string(),
                position: tokens.position(index),
            });
        }
        Self(token_positions)
    }
}

impl std::fmt::Display for TokenizedQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (index, token) in self.0.iter().enumerate() {
            if index > 0 {
                write!(f, ", ")?;
            }
            write!(f, "({:?}, {})", token.text, token.position)?;
        }
        write!(f, "]")
    }
}

pub(super) fn record_tokenized_query(snapshot: &OnceLock<TokenizedQuery>, tokens: &Tokens) {
    snapshot.get_or_init(|| TokenizedQuery::from_tokens(tokens));
}

pub(super) fn fmt_tokenized_query(
    snapshot: &OnceLock<TokenizedQuery>,
    separator: &str,
    f: &mut std::fmt::Formatter<'_>,
) -> std::fmt::Result {
    if let Some(tokens) = snapshot.get() {
        write!(f, "{separator}tokenized_query={tokens}")?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TokenizedLeafKind {
    Match,
    Phrase,
}

impl std::fmt::Display for TokenizedLeafKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Match => write!(f, "Match"),
            Self::Phrase => write!(f, "Phrase"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct TokenizedQueryLeaf {
    kind: TokenizedLeafKind,
    column: Option<String>,
    tokens: TokenizedQuery,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct TokenizedCompoundQuery(Vec<TokenizedQueryLeaf>);

impl std::fmt::Display for TokenizedCompoundQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (index, leaf) in self.0.iter().enumerate() {
            if index > 0 {
                write!(f, ", ")?;
            }
            write!(
                f,
                "{}(column={:?}, tokens={})",
                leaf.kind,
                leaf.column.as_deref().unwrap_or_default(),
                leaf.tokens
            )?;
        }
        write!(f, "]")
    }
}

pub(super) fn fmt_tokenized_compound_query(
    snapshot: &OnceLock<TokenizedCompoundQuery>,
    separator: &str,
    f: &mut std::fmt::Formatter<'_>,
) -> std::fmt::Result {
    if let Some(tokens) = snapshot.get() {
        write!(f, "{separator}tokenized_query={tokens}")?;
    }
    Ok(())
}

/// Fall back to the default simple tokenizer when no on-disk FTS segment exists.
pub(super) fn default_text_tokenizer() -> Box<dyn LanceTokenizer> {
    Box::new(TextTokenizer::new(
        TextAnalyzer::builder(SimpleTokenizer::default()).build(),
    ))
}

pub(super) fn tokenizer_for_match_query(
    index: &InvertedIndex,
    fuzziness: Option<u32>,
) -> Box<dyn LanceTokenizer> {
    // Preserve the legacy explicit-fuzzy behavior, while AUTO fuzziness uses
    // the index analyzer so its source terms share the indexed vocabulary's
    // normalization and filtering.
    if !matches!(fuzziness, Some(distance) if distance > 0) {
        return index.tokenizer();
    }

    let analyzer = TextAnalyzer::from(SimpleTokenizer::default());
    match index.tokenizer().doc_type() {
        DocType::Text => Box::new(TextTokenizer::new(analyzer)),
        DocType::Json => Box::new(JsonTokenizer::new(analyzer)),
    }
}

pub(super) fn tokenize_compound_query(
    query: &FtsQuery,
    index: &InvertedIndex,
) -> Result<TokenizedCompoundQuery> {
    fn visit(
        query: &FtsQuery,
        index: &InvertedIndex,
        leaves: &mut Vec<TokenizedQueryLeaf>,
    ) -> Result<()> {
        match query {
            FtsQuery::Match(query) => {
                let mut tokenizer = tokenizer_for_match_query(index, query.fuzziness);
                let tokens = try_collect_query_tokens(&query.terms, &mut tokenizer)?;
                leaves.push(TokenizedQueryLeaf {
                    kind: TokenizedLeafKind::Match,
                    column: query.column.clone(),
                    tokens: TokenizedQuery::from_tokens(&tokens),
                });
            }
            FtsQuery::Phrase(query) => {
                let mut tokenizer = index.tokenizer();
                let tokens = try_collect_query_tokens(&query.terms, &mut tokenizer)?;
                leaves.push(TokenizedQueryLeaf {
                    kind: TokenizedLeafKind::Phrase,
                    column: query.column.clone(),
                    tokens: TokenizedQuery::from_tokens(&tokens),
                });
            }
            FtsQuery::Boost(query) => {
                visit(&query.positive, index, leaves)?;
                visit(&query.negative, index, leaves)?;
            }
            FtsQuery::MultiMatch(query) => {
                for query in &query.match_queries {
                    let mut tokenizer = tokenizer_for_match_query(index, query.fuzziness);
                    let tokens = try_collect_query_tokens(&query.terms, &mut tokenizer)?;
                    leaves.push(TokenizedQueryLeaf {
                        kind: TokenizedLeafKind::Match,
                        column: query.column.clone(),
                        tokens: TokenizedQuery::from_tokens(&tokens),
                    });
                }
            }
            // Unreachable: only `CompoundQueryExec` tokenizes this way, and
            // `supports_compound_scorer` rejects BM25F at any depth. The
            // combined_fields execs snapshot their own tokens.
            FtsQuery::CombinedFields(_) => {}
            FtsQuery::Boolean(query) => {
                for query in query
                    .should
                    .iter()
                    .chain(&query.must)
                    .chain(&query.must_not)
                {
                    visit(query, index, leaves)?;
                }
            }
        }
        Ok(())
    }

    let mut leaves = Vec::with_capacity(count_fts_leaves(query));
    visit(query, index, &mut leaves)?;
    Ok(TokenizedCompoundQuery(leaves))
}

pub(super) fn tokenize_cross_column_compound_query(
    query: &FtsQuery,
    indices: &HashMap<&str, &InvertedIndex>,
) -> Result<TokenizedCompoundQuery> {
    fn index_for_leaf<'a>(
        column: Option<&str>,
        kind: &str,
        indices: &HashMap<&str, &'a InvertedIndex>,
    ) -> Result<(&'a InvertedIndex, String)> {
        let column = column.ok_or_else(|| {
            Error::invalid_input(format!(
                "cross-column compound FTS {kind} leaf is missing its resolved column"
            ))
        })?;
        let index = indices.get(column).copied().ok_or_else(|| {
            Error::invalid_input(format!(
                "cross-column compound FTS has no opened index for {kind} column {column}"
            ))
        })?;
        Ok((index, column.to_string()))
    }

    fn visit(
        query: &FtsQuery,
        indices: &HashMap<&str, &InvertedIndex>,
        leaves: &mut Vec<TokenizedQueryLeaf>,
    ) -> Result<()> {
        match query {
            FtsQuery::Match(query) => {
                let (index, column) = index_for_leaf(query.column.as_deref(), "Match", indices)?;
                let mut tokenizer = tokenizer_for_match_query(index, query.fuzziness);
                let tokens = try_collect_query_tokens(&query.terms, &mut tokenizer)?;
                leaves.push(TokenizedQueryLeaf {
                    kind: TokenizedLeafKind::Match,
                    column: Some(column),
                    tokens: TokenizedQuery::from_tokens(&tokens),
                });
            }
            FtsQuery::Phrase(query) => {
                let (index, column) = index_for_leaf(query.column.as_deref(), "Phrase", indices)?;
                let mut tokenizer = index.tokenizer();
                let tokens = try_collect_query_tokens(&query.terms, &mut tokenizer)?;
                leaves.push(TokenizedQueryLeaf {
                    kind: TokenizedLeafKind::Phrase,
                    column: Some(column),
                    tokens: TokenizedQuery::from_tokens(&tokens),
                });
            }
            FtsQuery::Boost(query) => {
                visit(&query.positive, indices, leaves)?;
                visit(&query.negative, indices, leaves)?;
            }
            FtsQuery::MultiMatch(query) => {
                for query in &query.match_queries {
                    let (index, column) =
                        index_for_leaf(query.column.as_deref(), "MultiMatch", indices)?;
                    let mut tokenizer = tokenizer_for_match_query(index, query.fuzziness);
                    let tokens = try_collect_query_tokens(&query.terms, &mut tokenizer)?;
                    leaves.push(TokenizedQueryLeaf {
                        kind: TokenizedLeafKind::Match,
                        column: Some(column),
                        tokens: TokenizedQuery::from_tokens(&tokens),
                    });
                }
            }
            FtsQuery::Boolean(query) => {
                for query in query
                    .should
                    .iter()
                    .chain(&query.must)
                    .chain(&query.must_not)
                {
                    visit(query, indices, leaves)?;
                }
            }
            // Unreachable while `supports_compound_scorer` rejects BM25F; see
            // `compound_leaf_columns`.
            FtsQuery::CombinedFields(_) => {
                return Err(Error::invalid_input(
                    "cross-column compound FTS cannot take a combined_fields leaf".to_string(),
                ));
            }
        }
        Ok(())
    }

    let mut leaves = Vec::with_capacity(count_fts_leaves(query));
    visit(query, indices, &mut leaves)?;
    Ok(TokenizedCompoundQuery(leaves))
}
