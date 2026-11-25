// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Shared match preparation and scored-document batch representations.

use std::cmp::Ordering;
use std::sync::Arc;

use arrow::array::{AsArray, ListBuilder, UInt32Builder};
use arrow::datatypes::{Float32Type, UInt64Type};
use arrow_array::{Array, Float32Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, SchemaRef};
use lance_core::{Error, ROW_ID, Result};
use lance_index::scalar::inverted::builder::ScoredDoc;
use lance_index::scalar::inverted::query::{
    FtsQuery, FtsSearchParams, Operator, Tokens, uses_fuzzy_expansion,
};
use lance_index::scalar::inverted::{
    DOC_INDEX_COL, InvertedIndex, MemBM25Scorer, PreparedBm25Query, SCORE_COL, prepare_bm25_query,
};

use super::selection::FtsIndexMetrics;

#[derive(Clone)]
pub(super) struct PreparedMatch {
    pub(super) query: Arc<PreparedBm25Query>,
    pub(super) params: Arc<FtsSearchParams>,
    pub(super) operator: Operator,
}

impl PreparedMatch {
    pub(super) async fn new(
        indices: &[Arc<InvertedIndex>],
        tokens: Tokens,
        params: FtsSearchParams,
        operator: Operator,
        metrics: &FtsIndexMetrics,
        base_scorer: Option<Arc<MemBM25Scorer>>,
    ) -> Result<Self> {
        let query = Arc::new(
            prepare_bm25_query(indices, tokens, &params, Some(metrics), base_scorer).await?,
        );
        Ok(Self {
            query,
            params: Arc::new(params),
            operator,
        })
    }
}

pub(super) fn scored_documents_batch(
    schema: SchemaRef,
    documents: Vec<ScoredDoc>,
) -> Result<RecordBatch> {
    let row_ids = UInt64Array::from_iter_values(documents.iter().map(|document| document.row_id));
    let scores = Float32Array::from_iter_values(documents.iter().map(|document| document.score.0));
    let mut columns = vec![Arc::new(row_ids) as Arc<dyn Array>];
    if schema.field_with_name(DOC_INDEX_COL).is_ok() {
        let mut builder = ListBuilder::new(UInt32Builder::new()).with_field(Field::new(
            "item",
            DataType::UInt32,
            false,
        ));
        for document in &documents {
            builder.values().append_slice(&document.doc_index);
            builder.append(true);
        }
        columns.push(Arc::new(builder.finish()));
    }
    columns.push(Arc::new(scores));
    Ok(RecordBatch::try_new(schema, columns)?)
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct DocumentKey {
    row_id: u64,
    doc_index: Vec<u32>,
}

pub(super) fn batch_document_keys(batch: &RecordBatch) -> Result<Vec<DocumentKey>> {
    let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>();
    let doc_indices = batch
        .column_by_name(DOC_INDEX_COL)
        .map(|column| column.as_list::<i32>());
    (0..batch.num_rows())
        .map(|row| {
            let doc_index = if let Some(doc_indices) = doc_indices {
                if doc_indices.is_null(row) {
                    return Err(Error::internal(
                        "element-document FTS produced a null document coordinate".to_string(),
                    ));
                }
                doc_indices
                    .value(row)
                    .as_primitive::<arrow::datatypes::UInt32Type>()
                    .values()
                    .to_vec()
            } else {
                Vec::new()
            };
            Ok(DocumentKey {
                row_id: row_ids.value(row),
                doc_index,
            })
        })
        .collect()
}

pub(super) fn batch_scored_document_keys(batch: &RecordBatch) -> Result<Vec<(DocumentKey, f32)>> {
    let keys = batch_document_keys(batch)?;
    let scores = batch[SCORE_COL].as_primitive::<Float32Type>();
    Ok(keys
        .into_iter()
        .enumerate()
        .map(|(index, key)| (key, scores.value(index)))
        .collect())
}

pub(super) fn batch_scored_document_keys_sum_scores(
    batch: &RecordBatch,
) -> Result<Vec<(DocumentKey, f32)>> {
    let keys = batch_document_keys(batch)?;
    let schema = batch.schema();
    let score_columns = schema
        .fields()
        .iter()
        .enumerate()
        .filter(|(_, field)| field.name() == SCORE_COL)
        .map(|(index, _)| batch.column(index).as_primitive::<Float32Type>())
        .collect::<Vec<_>>();
    if score_columns.is_empty() {
        return Err(Error::internal(format!(
            "Boolean MUST result is missing required {SCORE_COL} columns"
        )));
    }
    keys.into_iter()
        .enumerate()
        .map(|(row, key)| {
            let score: f32 = score_columns.iter().map(|scores| scores.value(row)).sum();
            if !score.is_finite() {
                return Err(Error::internal(format!(
                    "Boolean MUST score sum must be finite, got {score} for row_id={}",
                    key.row_id
                )));
            }
            Ok((key, score))
        })
        .collect()
}

pub(super) fn document_key_scores_batch(
    schema: SchemaRef,
    values: impl IntoIterator<Item = (DocumentKey, f32)>,
) -> Result<RecordBatch> {
    scored_documents_batch(
        schema,
        values
            .into_iter()
            .map(|(key, score)| ScoredDoc::with_doc_index(key.row_id, key.doc_index, score))
            .collect(),
    )
}

pub(super) fn compare_scored_documents(
    (left_key, left_score): &(DocumentKey, f32),
    (right_key, right_score): &(DocumentKey, f32),
) -> Ordering {
    right_score
        .total_cmp(left_score)
        .then_with(|| left_key.cmp(right_key))
}

pub(super) fn count_fts_leaves(query: &FtsQuery) -> usize {
    match query {
        FtsQuery::Match(_) | FtsQuery::Phrase(_) => 1,
        FtsQuery::Boost(query) => {
            count_fts_leaves(&query.positive) + count_fts_leaves(&query.negative)
        }
        FtsQuery::MultiMatch(query) => query.match_queries.len(),
        // Unreachable while `supports_compound_scorer` rejects BM25F. Counted as
        // the per-column posting scans anyway, so the arm stays honest if that
        // changes.
        FtsQuery::CombinedFields(query) => query.column_names().count(),
        FtsQuery::Boolean(query) => query
            .should
            .iter()
            .chain(&query.must)
            .chain(&query.must_not)
            .map(count_fts_leaves)
            .sum(),
    }
}

/// Return every leaf column, including prohibited Boolean leaves.
///
/// The repeated, ordered list is useful beyond the distinct set returned by
/// `FtsQueryNode::columns`: each leaf contributes its own posting-partition
/// work and must use the tokenizer and statistics of its field.
pub(super) fn compound_leaf_columns(query: &FtsQuery) -> Result<Vec<&str>> {
    fn visit<'a>(query: &'a FtsQuery, columns: &mut Vec<&'a str>) -> Result<()> {
        let required_column = |column: &'a Option<String>, kind: &str| {
            column.as_deref().ok_or_else(|| {
                Error::invalid_input(format!(
                    "cross-column compound FTS {kind} leaf is missing its resolved column"
                ))
            })
        };

        match query {
            FtsQuery::Match(query) => columns.push(required_column(&query.column, "Match")?),
            FtsQuery::Phrase(query) => columns.push(required_column(&query.column, "Phrase")?),
            FtsQuery::Boost(query) => {
                visit(&query.positive, columns)?;
                visit(&query.negative, columns)?;
            }
            FtsQuery::MultiMatch(query) => {
                for query in &query.match_queries {
                    columns.push(required_column(&query.column, "MultiMatch")?);
                }
            }
            FtsQuery::Boolean(query) => {
                for query in query
                    .should
                    .iter()
                    .chain(&query.must)
                    .chain(&query.must_not)
                {
                    visit(query, columns)?;
                }
            }
            // Unreachable while `supports_compound_scorer` rejects BM25F: a
            // combined_fields leaf blends statistics across its columns, so it
            // cannot name one column per leaf the way this list requires.
            FtsQuery::CombinedFields(_) => {
                return Err(Error::invalid_input(
                    "cross-column compound FTS cannot take a combined_fields leaf".to_string(),
                ));
            }
        }
        Ok(())
    }

    let mut columns = Vec::with_capacity(count_fts_leaves(query));
    visit(query, &mut columns)?;
    Ok(columns)
}

pub(super) fn compound_query_uses_fuzzy_expansion(query: &FtsQuery) -> bool {
    match query {
        FtsQuery::Match(query) => uses_fuzzy_expansion(query.fuzziness),
        // combined_fields matches terms exactly; it has no fuzziness setting.
        FtsQuery::Phrase(_) | FtsQuery::CombinedFields(_) => false,
        FtsQuery::Boost(query) => {
            compound_query_uses_fuzzy_expansion(&query.positive)
                || compound_query_uses_fuzzy_expansion(&query.negative)
        }
        FtsQuery::MultiMatch(query) => query
            .match_queries
            .iter()
            .any(|query| uses_fuzzy_expansion(query.fuzziness)),
        FtsQuery::Boolean(query) => query
            .should
            .iter()
            .chain(&query.must)
            .chain(&query.must_not)
            .any(compound_query_uses_fuzzy_expansion),
    }
}
