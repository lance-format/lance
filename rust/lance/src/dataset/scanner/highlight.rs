// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::Scanner;
use crate::index::scalar::inverted::{ResolvedFtsField, resolve_fts_analyzer, resolve_fts_field};
use crate::io::exec::fts::default_text_tokenizer;
use crate::io::exec::highlight::HighlightExec;
use crate::{Error, Result};
use datafusion::physical_plan::ExecutionPlan;
use lance_core::datatypes::{LogicalType, OnMissing};
use lance_index::scalar::FullTextSearchQuery;
use lance_index::scalar::inverted::highlight::{HighlightLeaf, PreparedHighlightQuery};
use lance_index::scalar::inverted::query::{
    FtsQuery, FtsSearchParams, Operator, try_collect_query_tokens,
};
use lance_index::scalar::inverted::tokenizer::document_tokenizer::{
    DocType, LanceTokenizer, TextTokenizer,
};
use lance_tokenizer::{SimpleTokenizer, TextAnalyzer};
use std::collections::HashMap;
use std::sync::Arc;

struct PreparedField {
    source: ResolvedFtsField,
    analyzer: Option<Arc<dyn LanceTokenizer>>,
}
impl Scanner {
    pub(super) async fn highlight_plan(
        &self,
        input: Arc<dyn ExecutionPlan>,
        query: &FullTextSearchQuery,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let granularity = self.fts_document_granularity(&query.query)?;
        let canonical = |column: &str| {
            Ok(resolve_fts_field(self.dataset.schema(), column, granularity)?.canonical_path)
        };
        let query = canonicalize_fields(query.query.clone(), &canonical)?;
        let mut queries = Vec::new();
        collect_leaves(&query, &mut queries);
        let mut fields = HashMap::<String, PreparedField>::new();
        let mut leaves = Vec::with_capacity(queries.len());
        for leaf in queries {
            let (columns, text, operator, slop, params) = match &leaf {
                FtsQuery::Match(query) => (
                    vec![query.column.clone().ok_or_else(|| {
                        Error::internal("highlight Match column was not resolved")
                    })?],
                    query.terms.as_str(),
                    query.operator,
                    None,
                    FtsSearchParams::new()
                        .with_fuzziness(query.fuzziness)
                        .with_prefix_length(query.prefix_length),
                ),
                FtsQuery::Phrase(query) => (
                    vec![query.column.clone().ok_or_else(|| {
                        Error::internal("highlight Phrase column was not resolved")
                    })?],
                    query.terms.as_str(),
                    Operator::And,
                    Some(query.slop),
                    FtsSearchParams::new().with_fuzziness(Some(0)),
                ),
                FtsQuery::CombinedFields(query) => (
                    query.column_names().map(str::to_owned).collect(),
                    query.terms(),
                    query.operator(),
                    None,
                    FtsSearchParams::new().with_fuzziness(Some(0)),
                ),
                _ => return Err(Error::internal("highlight expected a lexical leaf")),
            };
            for column in &columns {
                if !fields.contains_key(column) {
                    let source = resolve_fts_field(self.dataset.schema(), column, granularity)?;
                    if self
                        .dataset
                        .schema()
                        .field_by_id(source.final_field_id)
                        .is_some_and(|field| field.logical_type == LogicalType::from("json"))
                    {
                        return Err(Error::not_supported_source(
                            format!("highlight does not support JSON source field '{column}'")
                                .into(),
                        ));
                    }
                    let analyzer = resolve_fts_analyzer(&self.dataset, column, granularity)
                        .await?
                        .map(Arc::from);
                    fields.insert(column.clone(), PreparedField { source, analyzer });
                }
            }
            // CombinedFields resolves one analyzer for the logical virtual document.
            // Other clauses retain their own field analyzers, independent of row coverage.
            let field = columns
                .iter()
                .filter_map(|column| fields.get(column))
                .find(|field| {
                    !matches!(leaf, FtsQuery::CombinedFields(_)) || field.analyzer.is_some()
                })
                .or_else(|| columns.first().and_then(|column| fields.get(column)))
                .ok_or_else(|| Error::internal("highlight query has no source analyzer"))?;
            let tokenizer = field
                .analyzer
                .clone()
                .unwrap_or_else(|| Arc::from(default_text_tokenizer()));
            if matches!(tokenizer.doc_type(), DocType::Json) {
                return Err(Error::not_supported_source(
                    "highlight does not support JSON analyzers".into(),
                ));
            }
            let mut query_tokenizer: Box<dyn LanceTokenizer> = if matches!(params.fuzziness, Some(distance) if distance > 0)
            {
                Box::new(TextTokenizer::new(TextAnalyzer::from(
                    SimpleTokenizer::default(),
                )))
            } else {
                tokenizer.box_clone()
            };
            let tokens = try_collect_query_tokens(text, &mut query_tokenizer)?;
            leaves.push(HighlightLeaf::try_new(
                columns, tokenizer, tokens, operator, slop, &params,
            )?);
        }
        let mut sources = fields
            .into_values()
            .map(|field| field.source)
            .collect::<Vec<_>>();
        sources.sort_by(|a, b| a.canonical_path.cmp(&b.canonical_path));
        let roots = sources
            .iter()
            .map(|source| lance_core::datatypes::format_field_path(&[source.root_column.as_str()]));
        let projection = self
            .dataset
            .empty_projection()
            .union_columns(roots, OnMissing::Error)?;
        let input = self.take(input, projection)?;
        Ok(Arc::new(HighlightExec::new(
            input,
            Arc::new(PreparedHighlightQuery::new(query, leaves)),
            Arc::new(sources),
        )))
    }
}

fn canonicalize_fields(
    mut query: FtsQuery,
    canonical: &impl Fn(&str) -> Result<String>,
) -> Result<FtsQuery> {
    match &mut query {
        FtsQuery::Match(query) => {
            query.column = query.column.as_deref().map(canonical).transpose()?
        }
        FtsQuery::Phrase(query) => {
            query.column = query.column.as_deref().map(canonical).transpose()?
        }
        FtsQuery::MultiMatch(query) => {
            for query in &mut query.match_queries {
                query.column = query.column.as_deref().map(canonical).transpose()?;
            }
        }
        FtsQuery::Boost(query) => {
            *query.positive = canonicalize_fields(*query.positive.clone(), canonical)?;
        }
        FtsQuery::Boolean(query) => {
            for child in query
                .must
                .iter_mut()
                .chain(&mut query.should)
                .chain(&mut query.must_not)
            {
                *child = canonicalize_fields(child.clone(), canonical)?;
            }
        }
        FtsQuery::CombinedFields(query) => {
            *query = lance_index::scalar::inverted::query::CombinedFieldsQuery::try_new(
                query.terms().to_owned(),
                query.column_names().map(canonical).collect::<Result<_>>()?,
            )?
            .with_operator(query.operator())
            .try_with_boosts(query.weighted_columns().map(|(_, weight)| weight).collect())?;
        }
    }
    Ok(query)
}

fn collect_leaves(query: &FtsQuery, leaves: &mut Vec<FtsQuery>) {
    match query {
        FtsQuery::Match(_) | FtsQuery::Phrase(_) | FtsQuery::CombinedFields(_) => {
            leaves.push(query.clone())
        }
        FtsQuery::MultiMatch(query) => {
            leaves.extend(query.match_queries.iter().cloned().map(FtsQuery::Match))
        }
        FtsQuery::Boost(query) => collect_leaves(&query.positive, leaves),
        FtsQuery::Boolean(query) => {
            for child in query
                .must
                .iter()
                .chain(&query.should)
                .chain(&query.must_not)
            {
                collect_leaves(child, leaves);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Dataset;
    use crate::dataset::{WriteMode, WriteParams};
    use crate::index::DatasetIndexExt;
    use arrow_array::{Array, RecordBatch, RecordBatchIterator, cast::AsArray, record_batch};
    use lance_index::IndexType;
    use lance_index::scalar::inverted::DocumentGranularity;
    use lance_index::scalar::inverted::InvertedIndexParams;
    use lance_index::scalar::inverted::highlight::HIGHLIGHTS_COL;
    use lance_index::scalar::inverted::query::{
        BooleanQuery, BoostQuery, CombinedFieldsQuery, MatchQuery, MultiMatchQuery, Occur,
        PhraseQuery,
    };
    use rstest::rstest;

    async fn write(batch: RecordBatch, params: Option<WriteParams>) -> Dataset {
        Dataset::write(
            RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
            "memory://",
            params,
        )
        .await
        .unwrap()
    }

    fn text_match(text: &str, field: &str) -> FtsQuery {
        MatchQuery::new(text.into())
            .with_column(Some(field.into()))
            .with_fuzziness(Some(0))
            .into()
    }

    fn hits(batch: &RecordBatch, row: usize) -> Vec<String> {
        let values = batch[HIGHLIGHTS_COL].as_list::<i32>().value(row);
        let values = values.as_struct();
        let fields = values.column_by_name("field").unwrap().as_string::<i32>();
        let coordinates = values
            .column_by_name("value_index")
            .unwrap()
            .as_list::<i32>();
        let matches = values.column_by_name("matches").unwrap().as_list::<i32>();
        (0..values.len())
            .flat_map(|i| {
                let mut value = batch[fields.value(i)].slice(row, 1);
                let path = coordinates.value(i);
                for ordinal in path
                    .as_primitive::<arrow_array::types::UInt32Type>()
                    .values()
                {
                    value = value.as_list::<i32>().value(0).slice(*ordinal as usize, 1);
                }
                let text = value.as_string::<i32>().value(0);
                let spans = matches.value(i);
                let spans = spans.as_struct();
                let starts = spans
                    .column_by_name("start")
                    .unwrap()
                    .as_primitive::<arrow_array::types::UInt64Type>();
                let ends = spans
                    .column_by_name("end")
                    .unwrap()
                    .as_primitive::<arrow_array::types::UInt64Type>();
                (0..spans.len())
                    .map(|j| text[starts.value(j) as usize..ends.value(j) as usize].to_owned())
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[rstest]
    #[case::flat(false)]
    #[case::indexed(true)]
    #[tokio::test]
    async fn query_shapes_preserve_results_and_hidden_projection(#[case] indexed: bool) {
        let batch = record_batch!(
            ("id", Int32, [0, 1, 2, 3]),
            ("title", Utf8, ["beta", "none", "beta", "alpha"]),
            (
                "body",
                Utf8,
                ["alpha", "alpha gamma", "new fast york running café", "beta"]
            )
        )
        .unwrap();
        let mut dataset = write(
            batch,
            Some(WriteParams {
                max_rows_per_file: 2,
                max_rows_per_group: 2,
                ..Default::default()
            }),
        )
        .await;
        if indexed {
            for field in ["title", "body"] {
                dataset
                    .create_index(
                        &[field],
                        IndexType::Inverted,
                        None,
                        &InvertedIndexParams::default().with_position(true),
                        true,
                    )
                    .await
                    .unwrap();
            }
        }
        let queries = vec![
            text_match("alpha", "body"),
            PhraseQuery::new("new york".into())
                .with_column(Some("body".into()))
                .with_slop(1)
                .into(),
            CombinedFieldsQuery::try_new("alpha beta".into(), vec!["title".into(), "body".into()])
                .unwrap()
                .with_operator(Operator::And)
                .into(),
            MultiMatchQuery::try_new("alpha".into(), vec!["title".into(), "body".into()])
                .unwrap()
                .into(),
            BoostQuery::new(
                text_match("alpha", "body"),
                text_match("gamma", "body"),
                None,
            )
            .into(),
            BooleanQuery::new([
                (
                    Occur::Should,
                    BooleanQuery::new([
                        (Occur::Must, text_match("alpha", "body")),
                        (Occur::Must, text_match("beta", "title")),
                    ])
                    .into(),
                ),
                (Occur::Should, text_match("gamma", "body")),
            ])
            .into(),
        ];
        for query in queries {
            let requires_index = matches!(query, FtsQuery::CombinedFields(_));
            let mut scan = dataset.scan();
            scan.full_text_search(FullTextSearchQuery::new_query(query))
                .unwrap()
                .project(&["id"])
                .unwrap()
                .with_row_id()
                .limit(Some(3), None)
                .unwrap();
            if !indexed && requires_index {
                let error = scan.try_into_batch().await.unwrap_err();
                assert!(matches!(error, Error::InvalidInput { .. }));
                assert!(error.to_string().contains("requires an inverted index"));
                continue;
            }
            let baseline = scan.try_into_batch().await.unwrap();
            scan.with_highlight();
            let highlighted = scan.try_into_batch().await.unwrap();
            assert_eq!(baseline.num_rows(), highlighted.num_rows());
            assert!(highlighted.column_by_name("body").is_none());
            for field in baseline.schema().fields() {
                assert_eq!(
                    baseline.column_by_name(field.name()),
                    highlighted.column_by_name(field.name())
                );
            }
            assert!(highlighted.column_by_name(HIGHLIGHTS_COL).is_some());
        }
    }

    #[tokio::test]
    async fn mixed_fuzzy_ranges_ignore_index_coverage() {
        let batch = record_batch!(("body", Utf8, ["bat cat"])).unwrap();
        let mut dataset = write(batch.clone(), None).await;
        dataset
            .create_index(
                &["body"],
                IndexType::Inverted,
                None,
                &InvertedIndexParams::default(),
                true,
            )
            .await
            .unwrap();
        dataset
            .append(
                RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
                Some(WriteParams {
                    mode: WriteMode::Append,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
        let query = MatchQuery::new("cat".into())
            .with_column(Some("body".into()))
            .with_fuzziness(Some(1))
            .with_max_expansions(1);
        let mut scan = dataset.scan();
        scan.full_text_search(FullTextSearchQuery::new_query(query.into()))
            .unwrap()
            .with_row_id();
        let baseline = scan.try_into_batch().await.unwrap();
        let highlighted = scan.with_highlight().try_into_batch().await.unwrap();
        assert_eq!(baseline.num_rows(), 2);
        assert_eq!(
            baseline.column_by_name("_score"),
            highlighted.column_by_name("_score")
        );
        let mut words = (0..highlighted.num_rows())
            .map(|row| hits(&highlighted, row))
            .collect::<Vec<_>>();
        words.sort();
        assert_eq!(words, vec![vec!["bat", "cat"], vec!["bat", "cat"]]);
    }

    #[tokio::test]
    async fn hidden_boolean_fields_filter_and_pagination() {
        let batch = record_batch!(
            ("id", Int32, [0, 1, 2]),
            ("title", Utf8, ["beta", "none", "beta"]),
            ("body", Utf8, ["alpha", "alpha gamma", "alpha"])
        )
        .unwrap();
        let dataset = write(batch, None).await;
        let query = BooleanQuery::new([
            (
                Occur::Should,
                BooleanQuery::new([
                    (Occur::Must, text_match("alpha", "body")),
                    (Occur::Must, text_match("beta", "title")),
                ])
                .into(),
            ),
            (Occur::Should, text_match("gamma", "body")),
        ]);
        let mut scan = dataset.scan();
        scan.full_text_search(FullTextSearchQuery::new_query(query.into()))
            .unwrap()
            .filter("id = 1")
            .unwrap()
            .project(&["id"])
            .unwrap();
        let highlighted = scan.with_highlight().try_into_batch().await.unwrap();
        assert_eq!(highlighted.num_rows(), 1);
        let values = highlighted[HIGHLIGHTS_COL].as_list::<i32>().value(0);
        let matches = values.as_struct()["matches"].as_list::<i32>().value(0);
        assert_eq!(
            matches.as_struct()["start"]
                .as_primitive::<arrow_array::types::UInt64Type>()
                .value(0),
            6
        );
        assert_eq!(
            matches.as_struct()["end"]
                .as_primitive::<arrow_array::types::UInt64Type>()
                .value(0),
            11
        );
        assert!(highlighted.column_by_name("body").is_none());
    }
    #[rstest]
    #[case::row(DocumentGranularity::Row)]
    #[case::element(DocumentGranularity::ListElement)]
    #[tokio::test]
    async fn nested_arrays_preserve_coordinates(#[case] granularity: DocumentGranularity) {
        use arrow_array::builder::{ListBuilder, StringBuilder};
        use arrow_schema::{Field, Schema};
        let mut values = ListBuilder::new(ListBuilder::new(StringBuilder::new()));
        values.values().values().append_value("new");
        values.values().values().append_null();
        values.values().values().append_value("york");
        values.values().append(true);
        values.values().values().append_value("new york");
        values.values().append(true);
        values.append(true);
        values.append(false);
        values.append(true);
        let array = Arc::new(values.finish());
        let schema = Arc::new(Schema::new(vec![Field::new(
            "body",
            array.data_type().clone(),
            true,
        )]));
        let batch = RecordBatch::try_new(schema, vec![array]).unwrap();
        let mut dataset = write(
            batch,
            Some(WriteParams {
                max_rows_per_file: 2,
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await;
        dataset
            .create_index(
                &["body"],
                IndexType::Inverted,
                None,
                &InvertedIndexParams::default()
                    .with_position(true)
                    .document_granularity(granularity),
                true,
            )
            .await
            .unwrap();
        let query = PhraseQuery::new("new york".into())
            .with_column(Some("body".into()))
            .with_document_granularity(granularity);
        let mut scan = dataset.scan();
        scan.full_text_search(FullTextSearchQuery::new_query(query.into()))
            .unwrap()
            .with_row_id();
        let baseline = scan.try_into_batch().await.unwrap();
        let highlighted = scan.with_highlight().try_into_batch().await.unwrap();
        for field in baseline.schema().fields() {
            assert_eq!(
                baseline.column_by_name(field.name()),
                highlighted.column_by_name(field.name())
            );
        }
        assert_eq!(highlighted.num_rows(), 1);
        let values = highlighted[HIGHLIGHTS_COL].as_list::<i32>().value(0);
        let coordinates = values
            .as_struct()
            .column_by_name("value_index")
            .unwrap()
            .as_list::<i32>();
        let coordinates = (0..coordinates.len())
            .map(|i| {
                coordinates
                    .value(i)
                    .as_primitive::<arrow_array::types::UInt32Type>()
                    .values()
                    .to_vec()
            })
            .collect::<Vec<_>>();
        if granularity == DocumentGranularity::Row {
            assert_eq!(coordinates, vec![vec![0, 0], vec![0, 2], vec![1, 0]]);
            assert_eq!(hits(&highlighted, 0), vec!["new", "york", "new", "york"]);
        } else {
            assert_eq!(coordinates, vec![vec![1, 0]]);
            assert_eq!(hits(&highlighted, 0), vec!["new", "york"]);
        }
    }

    #[rstest]
    #[case::physical(false)]
    #[case::stable(true)]
    #[tokio::test]
    async fn updates_deletes_and_pinned_snapshot(#[case] stable: bool) {
        use crate::dataset::UpdateBuilder;
        let batch = record_batch!(
            ("id", Int32, [0, 1, 2]),
            ("body", Utf8, ["old needle", "delete needle", "keep needle"])
        )
        .unwrap();
        let mut dataset = write(
            batch,
            Some(WriteParams {
                max_rows_per_file: 2,
                enable_stable_row_ids: stable,
                ..Default::default()
            }),
        )
        .await;
        dataset
            .create_index(
                &["body"],
                IndexType::Inverted,
                None,
                &InvertedIndexParams::default(),
                true,
            )
            .await
            .unwrap();
        let mut pinned = dataset.scan();
        pinned
            .full_text_search(FullTextSearchQuery::new_query(text_match("needle", "body")))
            .unwrap()
            .with_highlight();
        let result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("id = 0")
            .unwrap()
            .set("body", "'new needle'")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();
        let mut updated = result.new_dataset.as_ref().clone();
        updated.delete("id = 1").await.unwrap();
        let before = pinned.try_into_batch().await.unwrap();
        assert_eq!(before.num_rows(), 3);
        assert!(
            before["body"]
                .as_string::<i32>()
                .iter()
                .flatten()
                .any(|value| value == "old needle")
        );
        let mut scan = updated.scan();
        scan.full_text_search(FullTextSearchQuery::new_query(text_match("needle", "body")))
            .unwrap()
            .with_row_id();
        let baseline = scan.try_into_batch().await.unwrap();
        let highlighted = scan.with_highlight().try_into_batch().await.unwrap();
        assert_eq!(highlighted.num_rows(), 2);
        for field in baseline.schema().fields() {
            assert_eq!(
                baseline.column_by_name(field.name()),
                highlighted.column_by_name(field.name())
            );
        }
        for row in 0..highlighted.num_rows() {
            assert_eq!(hits(&highlighted, row), vec!["needle"]);
        }
    }

    #[tokio::test]
    async fn highlight_preserves_filtered_pagination() {
        let batch = record_batch!(
            ("id", Int32, [0, 1, 2]),
            (
                "body",
                Utf8,
                ["needle", "needle needle padding padding", "needle"]
            )
        )
        .unwrap();
        let dataset = write(batch, None).await;
        let mut scan = dataset.scan();
        scan.full_text_search(FullTextSearchQuery::new_query(text_match("needle", "body")))
            .unwrap()
            .filter("id != 1")
            .unwrap()
            .limit(Some(1), Some(1))
            .unwrap();
        let baseline = scan.try_into_batch().await.unwrap();
        let highlighted = scan.with_highlight().try_into_batch().await.unwrap();
        assert_eq!(highlighted.num_rows(), 1);
        assert_eq!(
            baseline.column_by_name("id"),
            highlighted.column_by_name("id")
        );
        assert_eq!(hits(&highlighted, 0), vec!["needle"]);
        let mut scan = dataset.scan();
        scan.full_text_search(FullTextSearchQuery::new_query(text_match("needle", "body")))
            .unwrap()
            .with_highlight()
            .filter("id = 1")
            .unwrap();
        let result = scan.try_into_batch().await.unwrap();
        assert_eq!(hits(&result, 0), vec!["needle", "needle"]);
    }

    #[rstest]
    #[case::stem("simple", "run cafe", "Running café", true, true)]
    #[case::code("code", "http", "myHTTPClient", false, false)]
    #[case::grapheme("simple", "cafe", "🙂 cafe\u{301}", false, false)]
    #[tokio::test]
    async fn persisted_analyzers_preserve_source(
        #[case] tokenizer: &str,
        #[case] query: &str,
        #[case] source: &str,
        #[case] stem: bool,
        #[case] fold: bool,
    ) {
        let batch = record_batch!(("body", Utf8, [source])).unwrap();
        let mut dataset = write(batch, None).await;
        let params = InvertedIndexParams::default()
            .base_tokenizer(tokenizer.to_owned())
            .split_identifiers(tokenizer == "code")
            .stem(stem)
            .ascii_folding(fold);
        dataset
            .create_index(&["body"], IndexType::Inverted, None, &params, true)
            .await
            .unwrap();
        let mut scan = dataset.scan();
        scan.full_text_search(FullTextSearchQuery::new_query(text_match(query, "body")))
            .unwrap()
            .with_highlight();
        let highlighted = scan.try_into_batch().await.unwrap();
        assert_eq!(highlighted.num_rows(), 1);
        let expected = if stem {
            vec!["Running", "café"]
        } else if tokenizer == "code" {
            vec!["HTTP"]
        } else {
            vec!["cafe\u{301}"]
        };
        assert_eq!(hits(&highlighted, 0), expected);
    }
    #[tokio::test]
    async fn combined_fields_reuses_indexed_analyzer_for_unindexed_fields() {
        let batch =
            record_batch!(("title", Utf8, ["Running"]), ("body", Utf8, ["unrelated"])).unwrap();
        let mut dataset = write(batch, None).await;
        dataset
            .create_index(
                &["body"],
                IndexType::Inverted,
                None,
                &InvertedIndexParams::default().stem(true),
                true,
            )
            .await
            .unwrap();
        let combined =
            CombinedFieldsQuery::try_new("run".into(), vec!["title".into(), "body".into()])
                .unwrap();
        let query = BooleanQuery::new([
            (Occur::Should, text_match("run", "title")),
            (Occur::Should, combined.into()),
        ]);
        let mut scan = dataset.scan();
        scan.full_text_search(FullTextSearchQuery::new_query(query.into()))
            .unwrap();
        let baseline = scan.try_into_batch().await.unwrap();
        assert_eq!(baseline.num_rows(), 1);
        let highlighted = scan.with_highlight().try_into_batch().await.unwrap();
        assert_eq!(hits(&highlighted, 0), vec!["Running"]);
        assert_eq!(
            highlighted.column_by_name("_score"),
            baseline.column_by_name("_score")
        );
    }
    #[tokio::test]
    async fn physical_layout_and_index_coverage_do_not_change_ranges() {
        let batch = record_batch!(
            ("id", Int32, [0, 1]),
            ("body", Utf8, ["bat cat", "bat cat"])
        )
        .unwrap();
        let mut dataset = write(
            batch,
            Some(WriteParams {
                max_rows_per_file: 1,
                max_rows_per_group: 1,
                ..Default::default()
            }),
        )
        .await;
        for stage in 0..4 {
            let mut scan = dataset.scan();
            scan.full_text_search(FullTextSearchQuery::new_query(
                MatchQuery::new("cat".into())
                    .with_column(Some("body".into()))
                    .with_fuzziness(Some(1))
                    .with_max_expansions(1)
                    .into(),
            ))
            .unwrap()
            .with_highlight();
            let result = scan.try_into_batch().await.unwrap();
            assert_eq!(result.num_rows(), if stage < 2 { 2 } else { 3 });
            for row in 0..result.num_rows() {
                assert_eq!(hits(&result, row), vec!["bat", "cat"], "stage {stage}");
            }
            match stage {
                0 => {
                    dataset
                        .create_index(
                            &["body"],
                            IndexType::Inverted,
                            None,
                            &InvertedIndexParams::default(),
                            true,
                        )
                        .await
                        .unwrap();
                }
                1 => {
                    let appended =
                        record_batch!(("id", Int32, [2]), ("body", Utf8, ["bat cat"])).unwrap();
                    dataset
                        .append(
                            RecordBatchIterator::new([Ok(appended.clone())], appended.schema()),
                            Some(WriteParams {
                                mode: WriteMode::Append,
                                ..Default::default()
                            }),
                        )
                        .await
                        .unwrap();
                }
                2 => {
                    crate::dataset::optimize::compact_files(
                        &mut dataset,
                        crate::dataset::optimize::CompactionOptions {
                            target_rows_per_fragment: 10,
                            ..Default::default()
                        },
                        None,
                    )
                    .await
                    .unwrap();
                    assert!(dataset.get_fragments().len() < 3);
                }
                _ => {}
            }
        }
    }

    #[tokio::test]
    async fn flat_highlight_respects_the_retrieval_analyzer() {
        let dataset = write(record_batch!(("body", Utf8, ["Rust run"])).unwrap(), None).await;
        let query = BooleanQuery::new([
            (Occur::Should, text_match("rust", "body")),
            (Occur::Should, text_match("run", "body")),
        ]);
        let mut scan = dataset.scan();
        scan.full_text_search(FullTextSearchQuery::new_query(query.into()))
            .unwrap()
            .with_highlight();
        let highlighted = scan.try_into_batch().await.unwrap();
        assert_eq!(highlighted.num_rows(), 1);
        assert_eq!(hits(&highlighted, 0), vec!["run"]);
    }
}
