// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Full-text-search execution nodes.
//!
//! Dataset-level query validation, granularity resolution, and fallback
//! planning live in `dataset::scanner`. This module owns physical execution:
//! segment access, query-shape nodes, tokenization, scoring coordination, and
//! result batching. Posting-list and ranking algorithms remain in `lance-index`.

mod boolean;
mod combined_fields;
mod compound;
mod corpus;
mod document;
mod flat_combined_fields;
mod flat_match;
mod match_query;
mod matches;
mod phrase;
mod query_tokens;
pub(crate) mod segments;
mod selection;

pub use boolean::{
    BoolSlot, BooleanQueryExec, BoostQueryExec, build_boolean_query_children,
    build_boolean_query_children_with_schema,
};
pub use combined_fields::CombinedFieldsQueryExec;
pub use compound::{CompoundQueryExec, CrossColumnCompoundQueryExec, HybridCompoundQueryExec};
pub(crate) use corpus::SharedFtsScorer;
pub use corpus::{FtsCorpusBinding, SharedFtsScorerExec};
pub use document::FtsDocumentExec;
pub use flat_combined_fields::{FlatCombinedFieldsExec, FlatStatsCoverage, FlatStatsScope};
pub use flat_match::{FlatMatchFilterExec, FlatMatchQueryExec, FtsFilterDocumentBinding};
pub use match_query::MatchQueryExec;
pub use phrase::PhraseQueryExec;
pub use selection::{FTS_SEGMENT_BIND_DURATION_METRIC, FtsIndexMetrics};

#[cfg(test)]
mod tests {

    use datafusion::common::tree_node::TreeNodeRecursion;
    use datafusion::physical_plan::ChildrenPropertiesMode;
    use datafusion::physical_plan::ReplaceChildrenOptions;
    use datafusion_physical_expr::PhysicalExpr;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use crate::index::DatasetIndexExt;
    use arrow::datatypes::{Float32Type, UInt64Type};
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int32Type;
    use arrow_array::{
        ArrayRef, Float32Array, Int32Array, RecordBatch, RecordBatchIterator, StringArray,
        UInt64Array, record_batch,
    };
    use arrow_schema::DataType;
    use datafusion::datasource::memory::MemorySourceConfig;
    use datafusion::error::{DataFusionError, Result as DataFusionResult};
    use datafusion::physical_plan::metrics::{ExecutionPlanMetricsSet, MetricBuilder, MetricsSet};
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use datafusion::physical_plan::{
        DisplayAs, DisplayFormatType, PlanProperties, SendableRecordBatchStream,
    };
    use datafusion::{execution::TaskContext, physical_plan::ExecutionPlan};
    use futures::{TryStreamExt, stream};
    use lance_core::utils::futures::StreamOnDropExt;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_core::{ROW_ID, utils::address::RowAddress};
    use lance_datafusion::datagen::DatafusionDatagenExt;
    use lance_datafusion::exec::{ExecutionStatsCallback, ExecutionSummaryCounts};
    use lance_datafusion::utils::INDEX_SEGMENTS_SKIPPED_METRIC;
    use lance_datafusion::utils::{
        INDEX_CACHE_HITS_METRIC, INDEX_CACHE_MISSES_METRIC, PARTITIONS_SEARCHED_METRIC,
    };
    use lance_datagen::{BatchCount, ByteCount, RowCount};
    use lance_index::metrics::NoOpMetricsCollector;
    use lance_index::optimize::OptimizeOptions;
    use lance_index::scalar::inverted::query::{
        BooleanQuery, BoostQuery, CombinedFieldsQuery, FtsQuery, FtsSearchParams, MatchQuery,
        Occur, Operator, PhraseQuery, Tokens, collect_query_tokens, try_collect_query_tokens,
    };
    use lance_index::scalar::inverted::{
        CombinedCorpusStats, CombinedFieldColumn, DocumentGranularity, FTS_SCHEMA, InvertedIndex,
        InvertedListFormatVersion, Language, MemBM25Scorer, SCORE_COL, build_combined_bm25_scorer,
        build_global_bm25_scorer, prepare_bm25_query,
    };
    use lance_index::scalar::{FullTextSearchQuery, InvertedIndexParams};
    use lance_index::{IndexCriteria, IndexType};
    use lance_select::{
        RowAddrMask, RowAddrTreeMap,
        result::{IndexExprResult, IndexExprResultWireFormat},
    };
    use lance_table::format::IndexMetadata;
    use lance_table::io::manifest::read_manifest_indexes;
    use lance_table::system_index::frag_reuse::FRAG_REUSE_INDEX_NAME;
    use rstest::rstest;
    use uuid::Uuid;

    use crate::{
        Dataset,
        dataset::WriteParams,
        dataset::index::{DatasetIndexRemapperOptions, frag_reuse::cleanup_frag_reuse_index},
        dataset::optimize::{CompactionOptions, CompactionTask, TaskData, commit_compaction},
        dataset::{MergeInsertBuilder, WhenMatched, WhenNotMatched},
        index::loading::DatasetIndexInternalExt,
        io::exec::PreFilterSource,
        utils::test::{DatagenExt, FragmentCount, FragmentRowCount, NoContextTestFixture},
    };

    use super::corpus::SharedFtsScorerProducer;
    use super::query_tokens::tokenizer_for_match_query;
    use super::segments::open_fts_segments;
    use super::{
        BoolSlot, BoostQueryExec, CombinedFieldsQueryExec, CompoundQueryExec,
        CrossColumnCompoundQueryExec, FlatCombinedFieldsExec, FlatMatchQueryExec,
        FlatStatsCoverage, FlatStatsScope, HybridCompoundQueryExec, MatchQueryExec,
        PhraseQueryExec, SharedFtsScorer, SharedFtsScorerExec, build_boolean_query_children,
    };
    use crate::io::exec::utils::IndexMetrics;
    use datafusion::physical_plan::empty::EmptyExec;
    use datafusion::physical_plan::repartition::RepartitionExec;
    use datafusion::physical_plan::union::UnionExec;
    use datafusion_physical_plan::joins::HashJoinExec;
    use lance_datafusion::exec::OneShotExec;

    #[derive(Default)]
    pub(super) struct StatsHolder {
        collected_stats: Arc<Mutex<Option<ExecutionSummaryCounts>>>,
    }

    impl StatsHolder {
        pub(super) fn get_setter(&self) -> ExecutionStatsCallback {
            let collected_stats = self.collected_stats.clone();
            Arc::new(move |stats| {
                *collected_stats.lock().unwrap() = Some(stats.clone());
            })
        }

        pub(super) fn consume(self) -> ExecutionSummaryCounts {
            self.collected_stats.lock().unwrap().take().unwrap()
        }
    }

    pub(super) async fn create_segment_selection_fixture()
    -> (Arc<Dataset>, Vec<IndexMetadata>, Vec<u32>) {
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "text",
                lance_datagen::array::cycle_utf8_literals(&["quick brown fox"]),
            )
            .col(
                "other",
                lance_datagen::array::cycle_utf8_literals(&["not indexed"]),
            )
            .into_ram_dataset(FragmentCount::from(3), FragmentRowCount::from(2))
            .await
            .unwrap();
        let fragment_ids = dataset
            .get_fragments()
            .iter()
            .map(|fragment| fragment.id() as u32)
            .collect::<Vec<_>>();
        assert_eq!(fragment_ids.len(), 3);

        let params = InvertedIndexParams::default().with_position(true);
        let mut segments = Vec::with_capacity(fragment_ids.len());
        for fragment_id in &fragment_ids {
            let mut builder = dataset
                .create_index_builder(&["text"], IndexType::Inverted, &params)
                .name("segment_selection_fts".to_string())
                .fragments(vec![*fragment_id]);
            segments.push(builder.execute_uncommitted().await.unwrap());
        }
        dataset
            .commit_existing_index_segments("segment_selection_fts", "text", segments.clone())
            .await
            .unwrap();

        let committed = crate::index::scalar::inverted::load_segments(
            &dataset,
            "text",
            DocumentGranularity::Row,
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(committed.len(), fragment_ids.len());
        (Arc::new(dataset), committed, fragment_ids)
    }

    async fn create_hybrid_indexed_input_fixture(
        term: &str,
        limit: usize,
    ) -> (
        HybridCompoundQueryExec,
        Arc<MemBM25Scorer>,
        Arc<dyn ExecutionPlan>,
    ) {
        let (dataset, segments, _) = create_segment_selection_fixture().await;
        let mut dataset = dataset.as_ref().clone();
        let appended = lance_datagen::gen_batch()
            .col(
                "text",
                lance_datagen::array::cycle_utf8_literals(&["quick fresh", "fresh"]),
            )
            .col(
                "other",
                lance_datagen::array::cycle_utf8_literals(&["not indexed"]),
            )
            .into_reader_rows(RowCount::from(2), BatchCount::from(1));
        dataset.append(appended, None).await.unwrap();
        let tail = dataset.fragments().last().unwrap().clone();
        let dataset = Arc::new(dataset);
        let mut scanner = dataset.scan();
        scanner.project(&["text"]).unwrap();
        scanner.with_row_id().with_fragments(vec![tail]);
        let residual = scanner.create_plan().await.unwrap();
        let query = FtsQuery::Match(
            MatchQuery::new(term.to_string())
                .with_column(Some("text".to_string()))
                .with_fuzziness(Some(0)),
        );
        let params = FtsSearchParams::default().with_limit(Some(limit));
        let metrics = IndexMetrics::new(&ExecutionPlanMetricsSet::new(), 0);
        let indices = open_fts_segments(&dataset, "text", &segments, &metrics)
            .await
            .unwrap();
        let tokens = Tokens::new(vec![term.to_string()], indices[0].tokenizer().doc_type());
        let scorer = Arc::new(
            build_global_bm25_scorer(&indices, &tokens, &params, None)
                .await
                .unwrap(),
        );
        let indexed = Arc::new(
            CompoundQueryExec::new_with_segments(
                dataset.clone(),
                query.clone(),
                params.clone(),
                PreFilterSource::None,
                segments.clone(),
            )
            .with_base_scorer(scorer.clone()),
        );
        (
            HybridCompoundQueryExec::new(
                dataset,
                query,
                params,
                "text".to_string(),
                segments,
                residual,
            ),
            scorer,
            indexed,
        )
    }

    fn reconstruct_hybrid(
        exec: &HybridCompoundQueryExec,
        residual: Arc<dyn ExecutionPlan>,
    ) -> DataFusionResult<HybridCompoundQueryExec> {
        let rebuilt = HybridCompoundQueryExec::new(
            exec.dataset().clone(),
            exec.query().clone(),
            exec.params().clone(),
            exec.column().to_owned(),
            exec.segments().to_vec(),
            residual,
        );
        match exec.indexed_input() {
            Some((input, scorer)) => rebuilt.with_indexed_input(
                input.clone(),
                Arc::new(MemBM25Scorer::new(
                    scorer.total_tokens,
                    scorer.num_docs,
                    scorer.token_docs.clone(),
                )),
            ),
            None => Ok(rebuilt),
        }
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn test_hybrid_indexed_input_matches_native(
        #[values("quick", "fresh")] term: &str,
        #[values(1, 4)] limit: usize,
        #[values(false, true)] is_reconstructed: bool,
    ) {
        let (hybrid, scorer, indexed) = create_hybrid_indexed_input_fixture(term, limit).await;
        let expected = hybrid
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let replaced: Arc<dyn ExecutionPlan> =
            Arc::new(hybrid.with_indexed_input(indexed, scorer).unwrap());
        assert_eq!(replaced.children().len(), 2);
        assert_eq!(
            replaced
                .input_distribution_requirements()
                .per_child_distributions()
                .len(),
            2
        );
        // Scanner plans retain execution state, so the comparison needs an
        // independent residual scan over the same immutable dataset snapshot.
        let mut scanner = hybrid.dataset().scan();
        scanner.project(&[hybrid.column()]).unwrap();
        scanner
            .with_row_id()
            .with_fragments(vec![hybrid.dataset().fragments().last().unwrap().clone()]);
        let mut children = replaced.children().into_iter().cloned().collect::<Vec<_>>();
        children[0] = scanner.create_plan().await.unwrap();
        let rewritten = replaced
            .replace_children(
                children,
                ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
            )
            .unwrap();
        let rewritten = if is_reconstructed {
            let exec = rewritten.downcast_ref::<HybridCompoundQueryExec>().unwrap();
            Arc::new(reconstruct_hybrid(exec, exec.residual_input().clone()).unwrap())
                as Arc<dyn ExecutionPlan>
        } else {
            rewritten
        };
        let actual = rewritten
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(
            actual, expected,
            "indexed replacement must preserve scores and order"
        );
        assert!(actual.iter().map(RecordBatch::num_rows).sum::<usize>() > 0);
    }

    #[tokio::test]
    async fn test_hybrid_indexed_input_rejects_invalid_plan() {
        let (hybrid, scorer, indexed) = create_hybrid_indexed_input_fixture("quick", 3).await;
        let wrong_schema = Arc::new(EmptyExec::new(Arc::new(arrow_schema::Schema::empty())));
        let error = hybrid
            .with_indexed_input(wrong_schema, scorer.clone())
            .unwrap_err();
        assert!(matches!(error, DataFusionError::Plan(_)));
        assert!(error.to_string().contains("FTS schema"));

        let multiple_partitions =
            MemorySourceConfig::try_new_exec(&[vec![], vec![]], FTS_SCHEMA.clone(), None).unwrap();
        let error = hybrid
            .with_indexed_input(multiple_partitions, scorer.clone())
            .unwrap_err();
        assert!(matches!(error, DataFusionError::Plan(_)));
        assert!(error.to_string().contains("one partition"));

        let unbounded = HybridCompoundQueryExec::new(
            hybrid.dataset().clone(),
            hybrid.query().clone(),
            hybrid.params().clone().with_limit(None),
            hybrid.column().to_string(),
            hybrid.segments().to_vec(),
            hybrid.residual_input().clone(),
        );
        let error = unbounded.with_indexed_input(indexed, scorer).unwrap_err();
        assert!(matches!(error, DataFusionError::Plan(_)));
        assert!(error.to_string().contains("bounded limit"));

        let error = Arc::new(hybrid)
            .replace_children(
                vec![],
                ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
            )
            .unwrap_err();
        assert!(matches!(error, DataFusionError::Internal(_)));
        assert!(error.to_string().contains("expected 1 children"));
    }

    #[rstest::rstest]
    #[case::too_many_rows(false, "exceeded its bounded result limit")]
    #[case::null_score(true, "null row IDs or scores")]
    #[tokio::test]
    async fn test_hybrid_indexed_input_rejects_invalid_results(
        #[case] null_score: bool,
        #[case] message: &str,
        #[values(false, true)] is_reconstructed: bool,
    ) {
        let (hybrid, scorer, _) = create_hybrid_indexed_input_fixture("quick", 3).await;
        let scores = if null_score {
            vec![None, Some(1.0)]
        } else {
            vec![Some(1.0), Some(1.0)]
        };
        let batch = RecordBatch::try_new(
            FTS_SCHEMA.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![0, 1])),
                Arc::new(Float32Array::from(scores)),
            ],
        )
        .unwrap();
        let input = MemorySourceConfig::try_new_exec(
            &[vec![batch.clone(), batch]],
            FTS_SCHEMA.clone(),
            None,
        )
        .unwrap();
        let replaced: Arc<dyn ExecutionPlan> =
            Arc::new(hybrid.with_indexed_input(input, scorer).unwrap());
        let children = replaced.children().into_iter().cloned().collect();
        let rewritten = replaced
            .replace_children(
                children,
                ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
            )
            .unwrap();
        let rewritten = if is_reconstructed {
            let exec = rewritten.downcast_ref::<HybridCompoundQueryExec>().unwrap();
            Arc::new(reconstruct_hybrid(exec, exec.residual_input().clone()).unwrap())
                as Arc<dyn ExecutionPlan>
        } else {
            rewritten
        };
        let error = rewritten
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap_err();
        assert!(matches!(error, DataFusionError::Execution(_)));
        assert!(error.to_string().contains(message), "{error}");
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn test_hybrid_composition_state_preserves_corpus(
        #[values(false, true)] has_indexed_input: bool,
    ) {
        let (fixture, _, _) = create_hybrid_indexed_input_fixture("quick", 4).await;
        let query: FtsQuery = BooleanQuery::new([
            (
                Occur::Should,
                MatchQuery::new("quick".into())
                    .with_column(Some("text".into()))
                    .with_fuzziness(Some(0))
                    .into(),
            ),
            (
                Occur::Should,
                MatchQuery::new("fresh".into())
                    .with_column(Some("text".into()))
                    .with_fuzziness(Some(0))
                    .into(),
            ),
        ])
        .into();
        let metrics = IndexMetrics::new(&ExecutionPlanMetricsSet::new(), 0);
        let indices = open_fts_segments(fixture.dataset(), "text", fixture.segments(), &metrics)
            .await
            .unwrap();
        let tokens = Tokens::new(
            vec!["quick".into(), "fresh".into()],
            indices[0].tokenizer().doc_type(),
        );
        let committed = Arc::new(
            build_global_bm25_scorer(&indices, &tokens, fixture.params(), None)
                .await
                .unwrap(),
        );
        let selected = vec![fixture.segments()[1].clone()];
        let indexed: Arc<dyn ExecutionPlan> = Arc::new(
            CompoundQueryExec::new_with_segments(
                fixture.dataset().clone(),
                query.clone(),
                fixture.params().clone(),
                PreFilterSource::None,
                selected.clone(),
            )
            .with_base_scorer(committed.clone()),
        );
        let residual = |dataset: Arc<Dataset>| async move {
            let mut scan = dataset.scan();
            scan.project(&["text"])
                .unwrap()
                .with_row_id()
                .with_fragments(vec![dataset.fragments().last().unwrap().clone()]);
            scan.create_plan().await.unwrap()
        };
        let mut original = HybridCompoundQueryExec::new(
            fixture.dataset().clone(),
            query.clone(),
            fixture.params().clone(),
            "text".into(),
            selected.clone(),
            residual(fixture.dataset().clone()).await,
        );
        if has_indexed_input {
            original = original.with_indexed_input(indexed, committed).unwrap();
        }
        let restored =
            reconstruct_hybrid(&original, residual(fixture.dataset().clone()).await).unwrap();
        assert_eq!(restored.indexed_input().is_some(), has_indexed_input);
        let actual = execute_results(&restored).await.unwrap();
        assert_eq!(actual, execute_results(&original).await.unwrap());
        assert_eq!(actual.len(), 4);

        if has_indexed_input {
            // The worker searches one segment, but its supplied scorer describes
            // every committed segment. Dropping the pair changes real scores.
            let local = HybridCompoundQueryExec::new(
                fixture.dataset().clone(),
                query,
                fixture.params().clone(),
                "text".into(),
                selected,
                residual(fixture.dataset().clone()).await,
            );
            let mut local = execute_results(&local).await.unwrap();
            let mut actual = actual;
            local.sort_by_key(|row| row.0);
            actual.sort_by_key(|row| row.0);
            assert_eq!(
                local.iter().map(|row| row.0).collect::<Vec<_>>(),
                actual.iter().map(|row| row.0).collect::<Vec<_>>()
            );
            assert_ne!(
                actual, local,
                "a supplied corpus scorer must not become worker-local statistics"
            );
        }
    }

    fn tokenized_query_index_params() -> InvertedIndexParams {
        InvertedIndexParams::new("simple".to_string(), Language::English)
            .with_position(true)
            .lower_case(true)
            .stem(false)
            .remove_stop_words(true)
            .ascii_folding(false)
    }

    /// Builds a dataset where every column in `columns` holds "first and second"
    /// and carries its own inverted index. `combined_fields` needs an FTS index
    /// per target column, so the columns are indexed one at a time rather than
    /// as a single multi-column index.
    ///
    /// With `with_unindexed_append`, extra rows land outside the indices, which
    /// forces the mixed plan where an indexed exec and its flat sibling both run.
    async fn create_tokenized_query_fixture(
        columns: &[&str],
        with_unindexed_append: bool,
    ) -> Dataset {
        let corpus = || {
            columns
                .iter()
                .fold(lance_datagen::gen_batch(), |batch, column| {
                    batch.col(
                        *column,
                        lance_datagen::array::cycle_utf8_literals(&["first and second"]),
                    )
                })
        };

        let mut dataset = corpus()
            .into_ram_dataset(FragmentCount::from(1), FragmentRowCount::from(2))
            .await
            .unwrap();
        for column in columns {
            dataset
                .create_index(
                    &[*column],
                    IndexType::Inverted,
                    None,
                    &tokenized_query_index_params(),
                    true,
                )
                .await
                .unwrap();
        }

        if with_unindexed_append {
            let appended = corpus().into_reader_rows(RowCount::from(2), BatchCount::from(1));
            dataset.append(appended, None).await.unwrap();
        }
        dataset
    }

    fn find_plan_line<'a>(analysis: &'a str, node: &str) -> &'a str {
        analysis
            .lines()
            .find(|line| line.trim_start().starts_with(node))
            .unwrap_or_else(|| panic!("{node} missing from plan:\n{analysis}"))
    }

    pub(super) fn segment_uuid_for_fragment(segments: &[IndexMetadata], fragment_id: u32) -> Uuid {
        segments
            .iter()
            .find(|segment| {
                segment
                    .fragment_bitmap
                    .as_ref()
                    .is_some_and(|fragments| fragments.contains(fragment_id))
            })
            .map(|segment| segment.uuid)
            .unwrap()
    }

    pub(super) fn expected_row_ids(fragment_ids: &[u32]) -> Vec<u64> {
        let mut row_ids = fragment_ids
            .iter()
            .flat_map(|fragment_id| {
                (0..2).map(|offset| u64::from(RowAddress::new_from_parts(*fragment_id, offset)))
            })
            .collect::<Vec<_>>();
        row_ids.sort_unstable();
        row_ids
    }

    pub(super) async fn execute_results(
        plan: &dyn ExecutionPlan,
    ) -> DataFusionResult<Vec<(u64, f32)>> {
        let batches: Vec<RecordBatch> = plan
            .execute(0, Arc::new(TaskContext::default()))?
            .try_collect()
            .await?;
        let mut results = Vec::new();
        for batch in batches {
            let row_ids = batch[ROW_ID]
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            let scores = batch[SCORE_COL]
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap();
            results.extend(
                row_ids
                    .values()
                    .iter()
                    .copied()
                    .zip(scores.values().iter().copied()),
            );
        }
        results.sort_by_key(|(row_id, _)| *row_id);
        Ok(results)
    }

    pub(super) async fn execute_row_ids(plan: &dyn ExecutionPlan) -> DataFusionResult<Vec<u64>> {
        Ok(execute_results(plan)
            .await?
            .into_iter()
            .map(|(row_id, _)| row_id)
            .collect())
    }

    pub(super) fn metric_value(plan: &dyn ExecutionPlan, name: &str) -> usize {
        plan.metrics()
            .unwrap()
            .iter()
            .find(|metric| metric.value().name() == name)
            .unwrap()
            .value()
            .as_usize()
    }

    pub(super) fn assert_execution_error(error: DataFusionError, expected_message: &str) {
        assert!(
            matches!(&error, DataFusionError::Execution(_)),
            "expected execution error, got {error:?}"
        );
        assert!(
            error.to_string().contains(expected_message),
            "expected error containing {expected_message:?}, got {error}"
        );
    }

    #[tokio::test]
    async fn shared_fts_scorer_reports_cancelled_producer() {
        let scorer = Arc::new(SharedFtsScorer::<MemBM25Scorer>::new());
        let producer = SharedFtsScorerProducer::new(scorer.clone());
        drop(producer);

        let error = tokio::time::timeout(std::time::Duration::from_secs(1), scorer.wait())
            .await
            .expect("cancelled producer must wake scorer waiters")
            .unwrap_err();
        assert!(
            error.to_string().contains("producer was cancelled"),
            "{error}"
        );
    }

    pub(super) fn memory_input(batch: RecordBatch) -> Arc<dyn ExecutionPlan> {
        MemorySourceConfig::try_new_exec(&[vec![batch.clone()]], batch.schema(), None).unwrap()
    }

    pub(super) fn candidate_input(
        row_ids: Vec<u64>,
        is_scalar_index_query: bool,
    ) -> PreFilterSource {
        if is_scalar_index_query {
            let mask = RowAddrMask::from_allowed(RowAddrTreeMap::from_iter(row_ids));
            let batch = IndexExprResult::exact(mask)
                .serialize(
                    &roaring::RoaringBitmap::from_iter([0]),
                    IndexExprResultWireFormat::TwoMask,
                )
                .unwrap();
            PreFilterSource::ScalarIndexQuery(memory_input(batch))
        } else {
            PreFilterSource::FilteredRowIds(memory_input(
                record_batch!((ROW_ID, UInt64, row_ids)).unwrap(),
            ))
        }
    }

    pub(super) async fn indexed_match_dataset() -> Arc<Dataset> {
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "text",
                lance_datagen::array::cycle_utf8_literals(&["alpha", "beta"]),
            )
            .into_ram_dataset(FragmentCount::from(1), FragmentRowCount::from(2))
            .await
            .unwrap();
        dataset
            .create_index(
                &["text"],
                IndexType::Inverted,
                None,
                &InvertedIndexParams::default()
                    .stem(false)
                    .remove_stop_words(false),
                true,
            )
            .await
            .unwrap();
        Arc::new(dataset)
    }

    #[rstest]
    #[case::row_ids(false, "text")]
    #[case::row_ids_alias(false, "body")]
    #[case::selection_vector(true, "text")]
    #[case::selection_vector_alias(true, "body")]
    #[tokio::test]
    async fn test_9058_flat_candidates_preserve_corpus(
        #[case] is_scalar_index_query: bool,
        #[case] document_column: &str,
    ) {
        let dataset = indexed_match_dataset().await;
        let query = MatchQuery::new("alpha beta".to_string())
            .with_column(Some("text".to_string()))
            .with_document_granularity(DocumentGranularity::Row);
        let corpus = memory_input(
            record_batch!(
                (document_column, Utf8, ["alpha", "beta", "alpha"]),
                (ROW_ID, UInt64, [100, 101, 102])
            )
            .unwrap(),
        );
        let unfiltered = FlatMatchQueryExec::new_with_document_granularity(
            dataset.clone(),
            query.clone(),
            FtsSearchParams::default(),
            corpus.clone(),
            DocumentGranularity::Row,
            document_column.to_owned(),
        );
        assert!(!unfiltered.has_shared_scorer());
        let expected = execute_results(&unfiltered).await.unwrap();
        let shared = Arc::new(SharedFtsScorer::new());
        let filtered = Arc::new(
            FlatMatchQueryExec::new_with_document_granularity(
                dataset,
                query,
                FtsSearchParams::default(),
                corpus.clone(),
                DocumentGranularity::Row,
                document_column.to_owned(),
            )
            .with_shared_scorer(shared.clone())
            .with_candidate_filter(candidate_input(vec![101], is_scalar_index_query)),
        );

        assert!(filtered.has_shared_scorer());
        // An independent reconstruction must keep the candidate domain separate
        // from the corpus, including when the input column is aliased.
        let independent = FlatMatchQueryExec::new_with_document_granularity(
            filtered.dataset().clone(),
            filtered.query().clone(),
            filtered.params().clone(),
            corpus.clone(),
            filtered.document_granularity(),
            filtered.document_column().to_owned(),
        )
        .with_candidate_filter(filtered.candidate_filter().clone());
        assert_eq!(
            execute_results(&independent).await.unwrap(),
            vec![expected[1]]
        );
        assert_eq!(filtered.children().len(), 2);
        assert!(
            filtered
                .input_distribution_requirements()
                .per_child_distributions()
                .all(|distribution| {
                    matches!(
                        distribution,
                        datafusion_physical_expr::Distribution::SinglePartition
                    )
                })
        );
        assert_eq!(
            execute_results(filtered.as_ref()).await.unwrap(),
            vec![expected[1]]
        );
        assert_eq!(filtered.metrics().unwrap().output_rows(), Some(1));
        let scorer = shared.wait().await.unwrap();
        assert_eq!(scorer.num_docs(), 5);
        assert_eq!(scorer.num_docs_containing_token("alpha"), 3);
        assert_eq!(scorer.num_docs_containing_token("beta"), 2);

        // Replacing the candidate child must preserve its source kind. Empty
        // candidates still require the producer to collect the complete corpus.
        let empty = candidate_input(vec![], is_scalar_index_query);
        let expanded_corpus = memory_input(
            record_batch!(
                (document_column, Utf8, ["alpha", "beta", "alpha", "beta"]),
                (ROW_ID, UInt64, [100, 101, 102, 103])
            )
            .unwrap(),
        );
        let rebuilt = filtered
            .clone()
            .replace_children(
                vec![expanded_corpus, empty.execution_plan().unwrap().clone()],
                ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
            )
            .unwrap();
        assert!(execute_results(rebuilt.as_ref()).await.unwrap().is_empty());
        assert_eq!(rebuilt.metrics().unwrap().output_rows(), Some(0));
        assert_eq!(shared.wait().await.unwrap().num_docs(), 6);
        let error = filtered
            .replace_children(
                vec![],
                ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
            )
            .unwrap_err();
        assert!(matches!(error, DataFusionError::Internal(_)));
        assert!(error.to_string().contains("expected 2 children"));
    }

    #[test]
    fn test_9058_corpus_owner_rejects_missing_producer() {
        let input = Arc::new(EmptyExec::new(FTS_SCHEMA.clone()));
        let owner = Arc::new(SharedFtsScorerExec::new(
            input,
            Arc::new(SharedFtsScorer::new()),
        ));
        let Err(error) = owner.execute(0, Arc::new(TaskContext::default())) else {
            panic!("a scorer owner without a producer must fail before execution");
        };
        assert!(matches!(error, DataFusionError::Internal(_)));
        assert!(error.to_string().contains("got 0 and 0"));
        let error = owner
            .replace_children(
                vec![],
                ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
            )
            .unwrap_err();
        assert!(matches!(error, DataFusionError::Internal(_)));
        assert!(error.to_string().contains("requires one input"));
    }

    #[derive(Debug, Clone, Copy)]
    pub(super) enum ResidualFailure {
        Execute,
        Stream,
        Cancel,
    }

    /// A replayable source whose first execution fails or remains pending. Its
    /// retained metric handles also exercise the owner's metric de-duplication.
    #[derive(Debug, Clone)]
    pub(super) struct FailingResidualExec {
        pub(super) input: Arc<dyn ExecutionPlan>,
        pub(super) failure: ResidualFailure,
        pub(super) attempts: Arc<AtomicUsize>,
        pub(super) started: Arc<tokio::sync::Notify>,
        pub(super) dropped: Arc<AtomicBool>,
        pub(super) metrics: ExecutionPlanMetricsSet,
    }

    impl FailingResidualExec {
        pub(super) fn new(input: Arc<dyn ExecutionPlan>, failure: ResidualFailure) -> Self {
            Self {
                input,
                failure,
                attempts: Arc::new(AtomicUsize::new(0)),
                started: Arc::new(tokio::sync::Notify::new()),
                dropped: Arc::new(AtomicBool::new(false)),
                metrics: ExecutionPlanMetricsSet::new(),
            }
        }
    }

    impl DisplayAs for FailingResidualExec {
        fn fmt_as(&self, _: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "FailingResidual")
        }
    }

    impl ExecutionPlan for FailingResidualExec {
        fn name(&self) -> &str {
            "FailingResidualExec"
        }
        fn properties(&self) -> &Arc<PlanProperties> {
            self.input.properties()
        }
        fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
            vec![&self.input]
        }
        fn apply_expressions(
            &self,
            _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
        ) -> DataFusionResult<TreeNodeRecursion> {
            Ok(TreeNodeRecursion::Continue)
        }
        fn with_new_children(
            self: Arc<Self>,
            children: Vec<Arc<dyn ExecutionPlan>>,
        ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
            assert_eq!(children.len(), 1);
            Ok(Arc::new(Self {
                input: children[0].clone(),
                ..self.as_ref().clone()
            }))
        }
        fn metrics(&self) -> Option<MetricsSet> {
            Some(self.metrics.clone_inner())
        }
        fn execute(
            &self,
            partition: usize,
            context: Arc<TaskContext>,
        ) -> DataFusionResult<SendableRecordBatchStream> {
            MetricBuilder::new(&self.metrics)
                .counter("residual_executions", partition)
                .add(1);
            if self.attempts.fetch_add(1, Ordering::SeqCst) != 0 {
                return self.input.execute(partition, context);
            }
            match self.failure {
                ResidualFailure::Execute => Err(DataFusionError::Execution(
                    "injected residual execute failure".to_string(),
                )),
                ResidualFailure::Stream => Ok(Box::pin(RecordBatchStreamAdapter::new(
                    self.schema(),
                    stream::once(async {
                        Err(DataFusionError::Execution(
                            "injected residual stream failure".to_string(),
                        ))
                    }),
                ))),
                ResidualFailure::Cancel => {
                    let started = self.started.clone();
                    let dropped = self.dropped.clone();
                    let stream = stream::once(async move {
                        started.notify_one();
                        futures::future::pending::<DataFusionResult<RecordBatch>>().await
                    })
                    .on_drop(move || dropped.store(true, Ordering::SeqCst));
                    Ok(Box::pin(RecordBatchStreamAdapter::new(
                        self.schema(),
                        stream,
                    )))
                }
            }
        }
    }

    #[rstest]
    #[case::execute(ResidualFailure::Execute, "injected residual execute failure")]
    #[case::stream(ResidualFailure::Stream, "injected residual stream failure")]
    #[tokio::test]
    async fn test_9058_flat_publishes_input_error(
        #[case] failure: ResidualFailure,
        #[case] message: &str,
    ) {
        let dataset = indexed_match_dataset().await;
        let input = Arc::new(FailingResidualExec::new(
            memory_input(
                record_batch!(("text", Utf8, ["alpha"]), (ROW_ID, UInt64, [100])).unwrap(),
            ),
            failure,
        ));
        let shared = Arc::new(SharedFtsScorer::new());
        let flat = FlatMatchQueryExec::new(
            dataset,
            MatchQuery::new("alpha".to_string())
                .with_column(Some("text".to_string()))
                .with_document_granularity(DocumentGranularity::Row),
            FtsSearchParams::default(),
            input,
        )
        .unwrap()
        .with_shared_scorer(shared.clone());
        assert_execution_error(execute_results(&flat).await.unwrap_err(), message);
        let error = tokio::time::timeout(Duration::from_secs(1), shared.wait())
            .await
            .unwrap()
            .unwrap_err();
        assert_execution_error(error, message);
    }

    #[test]
    fn execute_without_context() {
        // These tests ensure we can create nodes and call execute without a tokio Runtime
        // being active.  This is a requirement for proper implementation of a Datafusion foreign
        // table provider.
        let fixture = NoContextTestFixture::new();
        let match_query = MatchQueryExec::new(
            Arc::new(fixture.dataset.clone()),
            MatchQuery::new("blah".to_string())
                .with_column(Some("text".to_string()))
                .with_document_granularity(DocumentGranularity::Row),
            FtsSearchParams::default(),
            PreFilterSource::None,
        )
        .unwrap();
        match_query
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap();
        let metrics = match_query.metrics().unwrap();
        assert!(metrics.elapsed_compute().unwrap() > 0);

        let flat_input = lance_datagen::gen_batch()
            .col(
                "text",
                lance_datagen::array::rand_utf8(ByteCount::from(10), false),
            )
            .into_df_exec(RowCount::from(15), BatchCount::from(2));

        let flat_match_query = FlatMatchQueryExec::new(
            Arc::new(fixture.dataset.clone()),
            MatchQuery::new("blah".to_string())
                .with_column(Some("text".to_string()))
                .with_document_granularity(DocumentGranularity::Row),
            FtsSearchParams::default(),
            flat_input,
        )
        .unwrap();
        flat_match_query
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap();
        let metrics = flat_match_query.metrics().unwrap();
        assert!(metrics.elapsed_compute().unwrap() > 0);

        let phrase_query = PhraseQueryExec::new(
            Arc::new(fixture.dataset.clone()),
            PhraseQuery::new("blah".to_string())
                .with_document_granularity(DocumentGranularity::Row),
            FtsSearchParams::new().with_phrase_slop(Some(0)),
            PreFilterSource::None,
        )
        .unwrap();
        phrase_query
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap();
        let metrics = phrase_query.metrics().unwrap();
        assert!(metrics.elapsed_compute().unwrap() > 0);

        let boost_input_one = MatchQueryExec::new(
            Arc::new(fixture.dataset.clone()),
            MatchQuery::new("blah".to_string())
                .with_column(Some("text".to_string()))
                .with_document_granularity(DocumentGranularity::Row),
            FtsSearchParams::default(),
            PreFilterSource::None,
        )
        .unwrap();

        let boost_input_two = MatchQueryExec::new(
            Arc::new(fixture.dataset),
            MatchQuery::new("blah".to_string())
                .with_column(Some("text".to_string()))
                .with_document_granularity(DocumentGranularity::Row),
            FtsSearchParams::default(),
            PreFilterSource::None,
        )
        .unwrap();

        let boost_query = BoostQueryExec::new(
            BoostQuery::new(
                FtsQuery::Match(
                    MatchQuery::new("blah".to_string()).with_column(Some("text".to_string())),
                ),
                FtsQuery::Match(
                    MatchQuery::new("test".to_string()).with_column(Some("text".to_string())),
                ),
                Some(1.0),
            ),
            FtsSearchParams::default(),
            Arc::new(boost_input_one),
            Arc::new(boost_input_two),
        );
        boost_query
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap();
        let metrics = boost_query.metrics().unwrap();
        assert!(metrics.elapsed_compute().unwrap() > 0);
    }

    #[tokio::test]
    async fn test_analyze_plan_shows_indexed_and_flat_match_tokens() {
        let dataset = create_tokenized_query_fixture(&["text"], true).await;
        let query = MatchQuery::new("FIRST and SECOND".to_string())
            .with_column(Some("text".to_string()))
            .with_operator(Operator::And);
        let mut scanner = dataset.scan();
        scanner
            .full_text_search(FullTextSearchQuery::new_query(query.into()))
            .unwrap();

        let explained = scanner.explain_plan(false).await.unwrap();
        assert!(
            !explained.contains("tokenized_query="),
            "explain_plan should not claim runtime tokenization: {explained}"
        );

        let analysis = scanner.analyze_plan().await.unwrap();
        let expected = r#"tokenized_query=[("first", 0), ("second", 2)]"#;
        assert!(
            find_plan_line(&analysis, "MatchQuery:").contains(expected),
            "indexed MatchQuery is missing token positions:\n{analysis}"
        );
        assert!(
            find_plan_line(&analysis, "FlatMatchQuery:").contains(expected),
            "flat MatchQuery is missing token positions:\n{analysis}"
        );
    }

    #[tokio::test]
    async fn test_analyze_plan_shows_indexed_and_flat_phrase_tokens() {
        let dataset = create_tokenized_query_fixture(&["text"], true).await;
        let query =
            PhraseQuery::new("FIRST and SECOND".to_string()).with_column(Some("text".to_string()));
        let mut scanner = dataset.scan();
        scanner
            .full_text_search(FullTextSearchQuery::new_query(query.into()))
            .unwrap();

        let analysis = scanner.analyze_plan().await.unwrap();
        let expected = r#"tokenized_query=[("first", 0), ("second", 2)]"#;
        assert!(
            find_plan_line(&analysis, "PhraseQuery:").contains(expected),
            "indexed PhraseQuery is missing token positions:\n{analysis}"
        );
        assert!(
            find_plan_line(&analysis, "FlatMatchQuery:").contains(expected),
            "flat phrase path is missing token positions:\n{analysis}"
        );
    }

    #[tokio::test]
    async fn test_analyze_plan_shows_compound_leaf_tokens() {
        let dataset = create_tokenized_query_fixture(&["text"], false).await;
        let query = BooleanQuery::new([
            (
                Occur::Should,
                MatchQuery::new("FIRST and SECOND".to_string())
                    .with_column(Some("text".to_string()))
                    .into(),
            ),
            (
                Occur::Must,
                PhraseQuery::new("SECOND FIRST".to_string())
                    .with_column(Some("text".to_string()))
                    .with_slop(2)
                    .into(),
            ),
        ]);
        let mut scanner = dataset.scan();
        scanner
            .full_text_search(FullTextSearchQuery::new_query(query.into()))
            .unwrap();

        let analysis = scanner.analyze_plan().await.unwrap();
        let compound = find_plan_line(&analysis, "CompoundFtsScorer:");
        assert!(
            compound.contains(r#"Match(column="text", tokens=[("first", 0), ("second", 2)])"#),
            "compound Match leaf is missing token positions: {compound}"
        );
        assert!(
            compound.contains(r#"Phrase(column="text", tokens=[("second", 0), ("first", 1)])"#),
            "compound Phrase leaf is missing token positions: {compound}"
        );
    }

    /// A `combined_fields` scan is a leaf FTS node, so `analyze_plan` must show
    /// its tokens the way it does for `match` and `phrase`. Both children of a
    /// mixed plan report, or a `Boolean{combined_fields, match}` plan would show
    /// tokens on the `match` sibling only.
    #[tokio::test]
    async fn test_analyze_plan_shows_combined_fields_tokens() {
        let dataset = create_tokenized_query_fixture(&["title", "body"], true).await;
        let query = CombinedFieldsQuery::try_new(
            "FIRST and SECOND".to_string(),
            vec!["title".to_string(), "body".to_string()],
        )
        .unwrap();
        let mut scanner = dataset.scan();
        scanner
            .full_text_search(FullTextSearchQuery::new_query(query.into()))
            .unwrap();

        let explained = scanner.explain_plan(false).await.unwrap();
        assert!(
            !explained.contains("tokenized_query="),
            "explain_plan should not claim runtime tokenization: {explained}"
        );

        let analysis = scanner.analyze_plan().await.unwrap();
        let expected = r#"tokenized_query=[("first", 0), ("second", 2)]"#;
        assert!(
            find_plan_line(&analysis, "CombinedFieldsQuery:").contains(expected),
            "indexed combined_fields is missing token positions:\n{analysis}"
        );
        assert!(
            find_plan_line(&analysis, "FlatCombinedFields:").contains(expected),
            "flat combined_fields is missing token positions:\n{analysis}"
        );
    }

    #[tokio::test]
    async fn test_boolean_query_parts_searched_metrics() {
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "text",
                lance_datagen::array::cycle_utf8_literals(&["hello", "lance", "search"]),
            )
            .into_ram_dataset(FragmentCount::from(3), FragmentRowCount::from(5))
            .await
            .unwrap();

        dataset
            .create_index(
                &["text"],
                IndexType::Inverted,
                None,
                &InvertedIndexParams::default(),
                true,
            )
            .await
            .unwrap();

        let index_meta = dataset
            .load_scalar_index(IndexCriteria::default().for_column("text").supports_fts())
            .await
            .unwrap()
            .unwrap();
        let index = dataset
            .open_generic_index("text", &index_meta.uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        let inverted_index = index.as_any().downcast_ref::<InvertedIndex>().unwrap();
        let expected_parts = inverted_index.partition_count();

        let query = BooleanQuery::new([
            (
                Occur::Should,
                MatchQuery::new("hello".to_string())
                    .with_operator(Operator::And)
                    .into(),
            ),
            (
                Occur::Must,
                MatchQuery::new("lance".to_string())
                    .with_operator(Operator::And)
                    .into(),
            ),
        ]);
        let expected_total = expected_parts * 2;

        let mut scanner = dataset.scan();
        scanner
            .project(&["text"])
            .unwrap()
            .with_row_id()
            .full_text_search(FullTextSearchQuery::new_query(query.into()))
            .unwrap();
        let analysis = scanner.analyze_plan().await.unwrap();
        let compound_line = analysis
            .lines()
            .find(|line| line.contains("CompoundFtsScorer"))
            .unwrap();
        assert!(
            compound_line.contains(&format!("{PARTITIONS_SEARCHED_METRIC}={expected_total}")),
            "compound FTS scorer metrics missing partitions_searched: {compound_line}"
        );
    }

    /// The cross-field scorer build reads one posting-metadata row per
    /// (term, column, partition), and both `combined_fields` execs must report
    /// those to their query's index-cache counters. Unreported, `EXPLAIN ANALYZE`
    /// on a cold cross-field query undercounts `index_cache_misses`, so its
    /// `index_cache_hit_ratio` is not comparable with `match`'s on the same data.
    #[tokio::test]
    async fn test_combined_fields_scorer_build_reports_index_cache_metrics() {
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "title",
                lance_datagen::array::cycle_utf8_literals(&[
                    "hello world",
                    "lance search",
                    "hello lance",
                ]),
            )
            .col(
                "body",
                lance_datagen::array::cycle_utf8_literals(&["lance", "hello", "search hello"]),
            )
            .into_ram_dataset(FragmentCount::from(1), FragmentRowCount::from(9))
            .await
            .unwrap();
        for column in ["title", "body"] {
            dataset
                .create_index(
                    &[column],
                    IndexType::Inverted,
                    None,
                    &InvertedIndexParams::default(),
                    true,
                )
                .await
                .unwrap();
        }
        let dataset = Arc::new(dataset);

        let query = CombinedFieldsQuery::try_new(
            "hello lance".to_string(),
            vec!["title".to_string(), "body".to_string()],
        )
        .unwrap();
        let params = FtsSearchParams::default().with_limit(Some(10));

        // Open the same segments the execs will, both to size the expected lookup
        // count and to hand the preset runs a scorer identical to the one they
        // would have built.
        let mut columns = Vec::new();
        for (column, weight) in query.weighted_columns() {
            let segments = crate::index::scalar::inverted::load_segments(
                &dataset,
                column,
                DocumentGranularity::Row,
            )
            .await
            .unwrap()
            .expect("FTS index just created");
            let metrics_set = ExecutionPlanMetricsSet::new();
            let indices = open_fts_segments(
                &dataset,
                column,
                &segments,
                &IndexMetrics::new(&metrics_set, 0),
            )
            .await
            .unwrap();
            columns.push(CombinedFieldColumn {
                column: column.to_string(),
                weight,
                indices,
                stale_rows: None,
            });
        }
        let mut tokenizer = columns[0].indices[0].tokenizer();
        let tokens = collect_query_tokens(query.terms(), &mut tokenizer);
        let mut terms = (0..tokens.len())
            .map(|index| tokens.get_token(index).to_string())
            .collect::<Vec<_>>();
        terms.sort_unstable();
        terms.dedup();
        let partitions: usize = columns
            .iter()
            .flat_map(|column| &column.indices)
            .map(|index| index.partition_count())
            .sum();
        // One posting-metadata row per (term, column, partition).
        let expected_lookups = terms.len() * partitions;
        assert!(expected_lookups > 0);
        let scorer = Arc::new(
            build_combined_bm25_scorer(&columns, &tokens, CombinedCorpusStats::IndexOnly, None)
                .await
                .unwrap(),
        );

        // Warm the index cache first: a cold run's counts depend on which entry
        // each lookup happens to load, while on a warm cache every lookup is a
        // hit and the two runs below differ only in the scorer build.
        let warmup = CombinedFieldsQueryExec::new(
            dataset.clone(),
            query.clone(),
            params.clone(),
            PreFilterSource::None,
        );
        let expected = execute_results(&warmup).await.unwrap();
        assert!(!expected.is_empty());

        let built = CombinedFieldsQueryExec::new(
            dataset.clone(),
            query.clone(),
            params.clone(),
            PreFilterSource::None,
        );
        assert_eq!(execute_results(&built).await.unwrap(), expected);
        let preset = CombinedFieldsQueryExec::new(
            dataset.clone(),
            query.clone(),
            params.clone(),
            PreFilterSource::None,
        )
        .with_base_scorer(scorer.clone());
        assert_eq!(
            execute_results(&preset).await.unwrap(),
            expected,
            "the preset scorer must match the one the exec builds",
        );
        assert_eq!(
            metric_value(&built, INDEX_CACHE_HITS_METRIC)
                - metric_value(&preset, INDEX_CACHE_HITS_METRIC),
            expected_lookups,
            "CombinedFieldsQueryExec must report the scorer build's cache lookups",
        );

        // The flat sibling builds its own scorer from the same per-column index
        // statistics. An empty input isolates those reads: the scanned rows
        // contribute none, so the count is absolute rather than a delta.
        let flat_schema = Arc::new(arrow_schema::Schema::new(vec![
            lance_core::ROW_ID_FIELD.clone(),
            arrow_schema::Field::new("title", DataType::Utf8, true),
            arrow_schema::Field::new("body", DataType::Utf8, true),
        ]));
        let num_columns = query.column_names().len();
        let flat = FlatCombinedFieldsExec::try_new(
            dataset.clone(),
            query.clone(),
            params.clone(),
            Arc::new(EmptyExec::new(flat_schema)),
            roaring::RoaringBitmap::new(),
            FlatStatsScope::PerColumn(vec![FlatStatsCoverage::default(); num_columns]),
            PreFilterSource::None,
        )
        .unwrap();
        assert!(execute_results(&flat).await.unwrap().is_empty());
        assert_eq!(
            metric_value(&flat, INDEX_CACHE_HITS_METRIC)
                + metric_value(&flat, INDEX_CACHE_MISSES_METRIC),
            expected_lookups,
            "FlatCombinedFieldsExec must report the scorer build's cache lookups",
        );
    }

    // Check the indexed leaf separately: the full scanner's fallback can hide stale
    // postings even when the addresses passed from the leaf to Take are invalid.
    async fn assert_partial_fts_results(
        dataset: &Dataset,
        index_uuid: Uuid,
        expected_indexed: &[i32],
        expected_all: &[i32],
    ) {
        let leaf = MatchQueryExec::new_with_segment_uuids(
            Arc::new(dataset.clone()),
            MatchQuery::new("searchable".into())
                .with_column(Some("text".into()))
                .with_document_granularity(DocumentGranularity::Row),
            FtsSearchParams::default().with_limit(Some(1000)),
            PreFilterSource::None,
            vec![index_uuid],
        )
        .unwrap();
        let mut addresses = execute_row_ids(&leaf).await.unwrap();
        assert_eq!(
            addresses.len(),
            expected_indexed.len(),
            "FTS must exclude postings outside its healed coverage"
        );
        assert!(
            addresses.iter().all(|&address| dataset
                .fragment_bitmap
                .contains(RowAddress::from(address).fragment_id())),
            "FTS must not return retired row addresses"
        );

        // Score order need not match physical row order. Verify Take preserves the
        // requested order and one output row per address, including with deletions.
        addresses.swap(0, 2);
        let mut expected_taken = expected_indexed.to_vec();
        expected_taken.swap(0, 2);
        let taken = Arc::new(dataset.clone())
            .take_builder(&addresses, Arc::new(dataset.schema().clone()))
            .unwrap()
            .with_row_address(false)
            .execute()
            .await
            .unwrap();
        assert_eq!(taken.num_rows(), addresses.len());
        assert_eq!(
            taken["id"].as_primitive::<Int32Type>().values().as_ref(),
            expected_taken
        );

        // The compacted destination must be scanned by the fallback. Exact IDs
        // detect both missing rows and duplicates across indexed and flat results.
        let mut scan = dataset.scan();
        scan.full_text_search(FullTextSearchQuery::new("searchable".into()))
            .unwrap();
        scan.limit(Some(1000), None).unwrap();
        let results = scan.try_into_batch().await.unwrap();
        let mut actual = results["id"].as_primitive::<Int32Type>().values().to_vec();
        actual.sort_unstable();
        assert_eq!(actual, expected_all);
    }

    #[rstest]
    #[case::no_deletion(false)]
    #[case::surviving_deletion(true)]
    #[tokio::test]
    async fn test_fts_v1_partial_coverage_after_fri_cleanup(#[case] has_surviving_deletion: bool) {
        let mut dataset = lance_datagen::gen_batch()
            .col("id", lance_datagen::array::step::<Int32Type>())
            .col(
                "text",
                lance_datagen::array::cycle_utf8_literals(&["searchable document"]),
            )
            .into_dataset_with_params(
                "memory://",
                FragmentCount::from(259),
                FragmentRowCount::from(2),
                Some(WriteParams {
                    max_rows_per_file: 2,
                    enable_stable_row_ids: false,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
        assert_eq!(dataset.get_fragments().len(), 259);
        if has_surviving_deletion {
            // Keep a deletion vector on a surviving indexed fragment, to cover the block-list path.
            dataset.delete("id = 202").await.unwrap();
        }
        let params = InvertedIndexParams::default()
            .format_version(InvertedListFormatVersion::V1)
            .with_position(false);
        // An existing index overlaps the rewrite, causing deferred compaction to
        // record reuse history while its replacement builds on a pinned snapshot.
        let old_index = dataset
            .create_index_builder(&["text"], IndexType::Inverted, &params)
            .name("text_idx".into())
            .fragments((0..3).chain(101..259).collect())
            .execute_uncommitted()
            .await
            .unwrap();
        dataset
            .commit_existing_index_segments("text_idx", "text", vec![old_index])
            .await
            .unwrap();
        let mut build_snapshot = dataset.clone();
        let mut parts = Vec::with_capacity(2);
        for fragments in [(0..130).collect(), (130..259).collect()] {
            parts.push(
                build_snapshot
                    .create_index_builder(&["text"], IndexType::Inverted, &params)
                    .name("text_idx".into())
                    .replace(true)
                    .fragments(fragments)
                    .execute_uncommitted()
                    .await
                    .unwrap(),
            );
        }
        assert!(parts.iter().all(|s| s.index_version == 1));

        let appended = lance_datagen::gen_batch()
            .col("id", lance_datagen::array::step_custom::<Int32Type>(518, 1))
            .col(
                "text",
                lance_datagen::array::cycle_utf8_literals(&["searchable document"]),
            )
            .into_reader_rows(RowCount::from(40), BatchCount::from(1));
        dataset
            .append(
                appended,
                Some(WriteParams {
                    max_rows_per_file: 2,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
        assert_eq!(dataset.get_fragments().len(), 279);

        // Rewrite 101 indexed fragments together with 20 appended fragments.
        // The staged build covers only 101/121 inputs, so it cannot cover the destination.
        let options = CompactionOptions {
            target_rows_per_fragment: 1000,
            defer_index_remap: true,
            ..Default::default()
        };
        let task = CompactionTask {
            task: TaskData {
                fragments: dataset
                    .fragments()
                    .iter()
                    .filter(|f| f.id < 101 || f.id >= 259)
                    .cloned()
                    .collect(),
                output_clustering_id: None,
                output_metadata: Default::default(),
            },
            read_version: dataset.version().version,
            options: options.clone(),
        };
        assert_eq!(task.task.fragments.len(), 121);
        let rewritten = task.execute(&dataset).await.unwrap();
        assert_eq!(rewritten.new_fragments.len(), 1);
        commit_compaction(
            &mut dataset,
            vec![rewritten],
            Arc::new(DatasetIndexRemapperOptions::default()),
            &options,
        )
        .await
        .unwrap();
        let stored = read_manifest_indexes(
            &dataset.object_store,
            &dataset.manifest_location,
            &dataset.manifest,
        )
        .await
        .unwrap();
        let fri_meta = stored
            .iter()
            .find(|i| i.name == FRAG_REUSE_INDEX_NAME)
            .unwrap();
        assert_eq!(fri_meta.index_version, 0);
        let fri = crate::index::frag_reuse::load_frag_reuse_index_details(&dataset, fri_meta)
            .await
            .unwrap();
        let group = &fri.versions.last().unwrap().groups[0];
        assert_eq!(group.old_frags.len(), 121);
        assert_eq!(group.new_frags.len(), 1);
        let new_fragment = group.new_frags[0].id;

        // Publish the staged build after compaction, from its original snapshot.
        let merged = build_snapshot
            .merge_existing_index_segments(parts)
            .await
            .unwrap();
        assert_eq!(merged.index_version, 1);
        let build_coverage = merged.fragment_bitmap.as_ref().unwrap();
        assert_eq!(build_coverage.len(), 259);
        assert_eq!(
            group
                .old_frags
                .iter()
                .filter(|f| build_coverage.contains(f.id as u32))
                .count(),
            101
        );
        build_snapshot
            .commit_existing_index_segments("text_idx", "text", vec![merged])
            .await
            .unwrap();
        dataset = build_snapshot;
        let raw = read_manifest_indexes(
            &dataset.object_store,
            &dataset.manifest_location,
            &dataset.manifest,
        )
        .await
        .unwrap();
        let raw_fts = raw.iter().find(|i| i.name == "text_idx").unwrap();
        let raw_overlap = group
            .old_frags
            .iter()
            .filter(|f| {
                raw_fts
                    .fragment_bitmap
                    .as_ref()
                    .unwrap()
                    .contains(f.id as u32)
            })
            .count();
        assert_eq!(raw_overlap, 101);
        let segments = crate::index::scalar::inverted::load_segments(
            &dataset,
            "text",
            DocumentGranularity::Row,
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(segments.len(), 1);
        let coverage = segments[0].fragment_bitmap.as_ref().unwrap();
        assert_eq!(coverage.len(), 158);
        assert!(!coverage.contains(new_fragment as u32));
        assert!(coverage.iter().all(|id| (101..259).contains(&id)));
        let index_uuid = segments[0].uuid;
        let expected_indexed = (202..518)
            .filter(|id| !has_surviving_deletion || *id != 202)
            .collect::<Vec<_>>();
        let expected_all = (0..558)
            .filter(|id| !has_surviving_deletion || *id != 202)
            .collect::<Vec<_>>();

        // A normal metadata commit persists the healed bitmap without rewriting
        // postings. Warm the index while the original FRI mapping still exists.
        dataset.delete("false").await.unwrap();
        let stored = read_manifest_indexes(
            &dataset.object_store,
            &dataset.manifest_location,
            &dataset.manifest,
        )
        .await
        .unwrap();
        let stored_fts = stored.iter().find(|i| i.name == "text_idx").unwrap();
        assert_eq!(stored_fts.uuid, index_uuid);
        assert_eq!(stored_fts.fragment_bitmap.as_ref(), Some(coverage));
        let before_cleanup = dataset.clone();
        let original_fri = dataset.frag_reuse_index().await.unwrap().unwrap();
        assert_eq!(original_fri.details.versions.len(), 1);
        assert_partial_fts_results(&dataset, index_uuid, &expected_indexed, &expected_all).await;

        // Once coverage no longer intersects the retired fragments or destination,
        // cleanup can discard this mapping. The unchanged postings must still be
        // restricted to live covered fragments, even when a deletion vector exists.
        cleanup_frag_reuse_index(&mut dataset).await.unwrap();
        let trimmed_fri = dataset.frag_reuse_index().await.unwrap().unwrap();
        assert!(trimmed_fri.details.versions.is_empty());
        assert_ne!(trimmed_fri.uuid, original_fri.uuid);
        assert_partial_fts_results(&dataset, index_uuid, &expected_indexed, &expected_all).await;
        assert_partial_fts_results(
            &before_cleanup,
            index_uuid,
            &expected_indexed,
            &expected_all,
        )
        .await;
    }

    #[rstest::rstest]
    #[case::all_fragments(false)]
    #[case::restricted_fragments(true)]
    #[tokio::test]
    async fn test_combined_fields_exec_opens_all_committed_segments(
        #[case] restrict_fragments: bool,
    ) {
        let (dataset, _segments, fragment_ids) = create_segment_selection_fixture().await;
        let query = CombinedFieldsQuery::try_new("quick".to_string(), vec!["text".to_string()])
            .unwrap()
            .try_with_boosts(vec![2.0])
            .unwrap();
        let params = FtsSearchParams::default().with_limit(Some(20));

        // One segment per fragment, so a scan that opens every committed segment
        // returns every fragment's matching row.
        let exec = CombinedFieldsQueryExec::new(
            dataset.clone(),
            query.clone(),
            params.clone(),
            PreFilterSource::None,
        );
        assert_eq!(
            execute_row_ids(&exec).await.unwrap(),
            expected_row_ids(&fragment_ids)
        );

        let selected_row_ids = expected_row_ids(&fragment_ids[..2]);
        let selected_row_count = selected_row_ids.len();
        let batch = arrow_array::record_batch!(("_rowid", UInt64, selected_row_ids)).unwrap();
        let source = PreFilterSource::FilteredRowIds(Arc::new(OneShotExec::from_batch(batch)));
        let mut filtered = CombinedFieldsQueryExec::new(dataset, query, params, source);
        let expected = if restrict_fragments {
            filtered = filtered
                .with_covered_fragments(roaring::RoaringBitmap::from_iter([fragment_ids[1]]));
            expected_row_ids(&fragment_ids[1..2])
        } else {
            expected_row_ids(&fragment_ids[..2])
        };
        assert_eq!(execute_row_ids(&filtered).await.unwrap(), expected);
        assert_eq!(metric_value(&filtered, "prefilter_loads"), 1);
        assert_eq!(
            metric_value(&filtered, "prefilter_input_rows"),
            selected_row_count
        );
        assert_eq!(
            metric_value(&filtered, "prefilter_row_ids"),
            selected_row_count
        );
    }

    #[tokio::test]
    async fn test_match_query_exec_with_base_scorer_matches_baseline() {
        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        // Skewed term distributions across two fragments — "lance" is common in
        // segment 0 and rare in segment 1 — so any local-IDF computation will
        // disagree with the global-IDF baseline. That makes the test sensitive
        // to a bug where `with_base_scorer` is silently ignored.
        let batches = vec![
            RecordBatch::try_from_iter(vec![
                ("id", Arc::new(Int32Array::from(vec![0, 1])) as ArrayRef),
                (
                    "text",
                    Arc::new(StringArray::from(vec![
                        Some("lance database"),
                        Some("lance search"),
                    ])) as ArrayRef,
                ),
            ])
            .unwrap(),
            RecordBatch::try_from_iter(vec![
                ("id", Arc::new(Int32Array::from(vec![2, 3])) as ArrayRef),
                (
                    "text",
                    Arc::new(StringArray::from(vec![
                        Some("lancd alpha"),
                        Some("lancd lance"),
                    ])) as ArrayRef,
                ),
            ])
            .unwrap(),
        ];
        let schema = batches[0].schema();
        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
        let mut ds = Dataset::write(
            reader,
            test_uri,
            Some(WriteParams {
                max_rows_per_file: 2,
                max_rows_per_group: 2,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let params = InvertedIndexParams::new("simple".to_string(), Language::English)
            .with_position(false)
            .lower_case(true)
            .stem(false)
            .remove_stop_words(true)
            .ascii_folding(false)
            .max_token_length(None);
        let fragment_ids = ds
            .get_fragments()
            .iter()
            .map(|fragment| fragment.id() as u32)
            .collect::<Vec<_>>();
        assert!(
            fragment_ids.len() >= 2,
            "test setup should produce >= 2 fragments, got {}",
            fragment_ids.len()
        );

        let mut metadatas = Vec::<IndexMetadata>::with_capacity(fragment_ids.len());
        for fragment_id in fragment_ids {
            let mut builder = ds
                .create_index_builder(&["text"], IndexType::Inverted, &params)
                .name("seg_fts".to_string())
                .fragments(vec![fragment_id]);
            metadatas.push(builder.execute_uncommitted().await.unwrap());
        }
        ds.commit_existing_index_segments("seg_fts", "text", metadatas.clone())
            .await
            .unwrap();
        assert_eq!(
            ds.load_indices_by_name("seg_fts").await.unwrap().len(),
            metadatas.len(),
            "expected one committed segment per fragment"
        );

        let dataset = Arc::new(ds);
        let query = MatchQuery::new("lance".to_string())
            .with_column(Some("text".to_string()))
            .with_document_granularity(DocumentGranularity::Row);
        let search_params = FtsSearchParams::default().with_limit(Some(10));

        // Baseline: the existing path that builds the global scorer locally.
        let baseline_exec = MatchQueryExec::new(
            dataset.clone(),
            query.clone(),
            search_params.clone(),
            PreFilterSource::None,
        )
        .unwrap();
        let baseline_batches: Vec<RecordBatch> = baseline_exec
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let baseline = concat_score_batches(&baseline_batches);
        assert!(
            !baseline.is_empty(),
            "baseline should return at least one hit"
        );

        // Override: build the global scorer manually via the public helper, then
        // construct the exec with the preset segments and the preset scorer.
        let preset_segments = crate::index::scalar::inverted::load_segments(
            &dataset,
            "text",
            DocumentGranularity::Row,
        )
        .await
        .unwrap()
        .expect("FTS index just created");
        let metrics_set = ExecutionPlanMetricsSet::new();
        let metrics = IndexMetrics::new(&metrics_set, 0);
        let indices = open_fts_segments(&dataset, "text", &preset_segments, &metrics)
            .await
            .unwrap();
        assert!(
            indices.len() >= 2,
            "expected >= 2 segments to exercise global IDF, got {}",
            indices.len()
        );

        let mut auto_tokenizer = tokenizer_for_match_query(&indices[0], None);
        let auto_tokens = collect_query_tokens("THE LANCE", &mut auto_tokenizer);
        assert_eq!(auto_tokens.len(), 1);
        assert_eq!(auto_tokens.get_token(0), "lance");
        let mut explicit_fuzzy_tokenizer = tokenizer_for_match_query(&indices[0], Some(1));
        let explicit_fuzzy_tokens =
            collect_query_tokens("THE LANCE", &mut explicit_fuzzy_tokenizer);
        assert_eq!(explicit_fuzzy_tokens.len(), 2);
        assert_eq!(explicit_fuzzy_tokens.get_token(0), "THE");
        assert_eq!(explicit_fuzzy_tokens.get_token(1), "LANCE");

        let auto_query = |terms: &str| {
            MatchQuery::new(terms.to_owned())
                .with_column(Some("text".to_owned()))
                .with_fuzziness(None)
                .with_document_granularity(DocumentGranularity::Row)
        };
        let lowercase_auto_exec = MatchQueryExec::new(
            dataset.clone(),
            auto_query("lance"),
            search_params.clone(),
            PreFilterSource::None,
        )
        .unwrap();
        let lowercase_auto_results = execute_results(&lowercase_auto_exec).await.unwrap();
        assert!(!lowercase_auto_results.is_empty());
        let normalized_auto_exec = MatchQueryExec::new(
            dataset.clone(),
            auto_query("THE LANCE"),
            search_params.clone(),
            PreFilterSource::None,
        )
        .unwrap();
        assert_eq!(
            execute_results(&normalized_auto_exec).await.unwrap(),
            lowercase_auto_results,
            "AUTO fuzzy Match must preserve index lowercase and stop-word analysis"
        );

        let mut tokenizer = indices[0].tokenizer();
        let tokens = try_collect_query_tokens(&query.terms, &mut tokenizer).unwrap();
        let global_scorer = Arc::new(
            build_global_bm25_scorer(&indices, &tokens, &search_params, None)
                .await
                .unwrap(),
        );

        let override_exec = MatchQueryExec::new_with_segments(
            dataset.clone(),
            query.clone(),
            search_params.clone(),
            PreFilterSource::None,
            preset_segments.clone(),
        )
        .unwrap()
        .with_base_scorer(global_scorer);
        let override_batches: Vec<RecordBatch> = override_exec
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let overridden = concat_score_batches(&override_batches);

        assert_eq!(
            baseline.len(),
            overridden.len(),
            "row count differs: baseline={}, override={}",
            baseline.len(),
            overridden.len()
        );
        for (i, (b, o)) in baseline.iter().zip(overridden.iter()).enumerate() {
            assert_eq!(
                b.0, o.0,
                "row id mismatch at rank {}: baseline={}, override={}",
                i, b.0, o.0
            );
            assert_eq!(
                b.1, o.1,
                "score mismatch at rank {} (row id {}): baseline={}, override={}",
                i, b.0, b.1, o.1
            );
        }

        // Sanity check on FTS schema before extracting columns above.
        for batch in baseline_batches.iter().chain(override_batches.iter()) {
            assert!(
                batch.column_by_name(ROW_ID).is_some(),
                "FTS output is expected to carry a row id column"
            );
            assert_eq!(
                batch.column_by_name(SCORE_COL).unwrap().data_type(),
                &DataType::Float32,
                "FTS score column should be Float32"
            );
        }

        // A distributed fuzzy subset must receive the canonical vocabulary
        // together with its scorer. A scorer-only override cannot reproduce a
        // globally capped rewrite from worker-local segment vocabularies.
        let fuzzy_query = MatchQuery::new("lancx".to_string())
            .with_column(Some("text".to_string()))
            .with_fuzziness(Some(1))
            .with_max_expansions(1)
            .with_document_granularity(DocumentGranularity::Row);
        let fuzzy_params = search_params
            .clone()
            .with_fuzziness(Some(1))
            .with_max_expansions(1);
        let mut tokenizer = tokenizer_for_match_query(&indices[0], fuzzy_query.fuzziness);
        let fuzzy_tokens = collect_query_tokens(&fuzzy_query.terms, &mut tokenizer);
        let prepared = Arc::new(
            prepare_bm25_query(&indices, fuzzy_tokens, &fuzzy_params, None, None)
                .await
                .unwrap(),
        );
        assert_eq!(prepared.tokens().len(), 1);
        assert_eq!(prepared.tokens().get_token(0), "lancd");

        let prepared_full_exec = MatchQueryExec::new_with_segments(
            dataset.clone(),
            fuzzy_query.clone(),
            search_params.clone(),
            PreFilterSource::None,
            preset_segments.clone(),
        )
        .unwrap()
        .with_prepared_query(prepared.clone());
        let prepared_full_results = execute_row_ids(&prepared_full_exec).await.unwrap();
        assert_eq!(prepared_full_results.len(), 2);

        let all_committed_scorer_exec = MatchQueryExec::new(
            dataset.clone(),
            fuzzy_query.clone(),
            search_params.clone(),
            PreFilterSource::None,
        )
        .unwrap()
        .with_base_scorer(prepared.scorer().clone());
        assert_eq!(
            execute_row_ids(&all_committed_scorer_exec).await.unwrap(),
            prepared_full_results
        );

        let explicit_full_scorer_exec = MatchQueryExec::new_with_segments(
            dataset.clone(),
            fuzzy_query.clone(),
            search_params.clone(),
            PreFilterSource::None,
            preset_segments.clone(),
        )
        .unwrap()
        .with_base_scorer(prepared.scorer().clone());
        assert_eq!(
            execute_row_ids(&explicit_full_scorer_exec).await.unwrap(),
            prepared_full_results
        );

        let subset_exec = MatchQueryExec::new_with_segments(
            dataset.clone(),
            fuzzy_query.clone(),
            search_params.clone(),
            PreFilterSource::None,
            vec![preset_segments[0].clone()],
        )
        .unwrap()
        .with_prepared_query(prepared.clone());
        assert!(execute_row_ids(&subset_exec).await.unwrap().is_empty());

        let scorer_only_exec = MatchQueryExec::new_with_segments(
            dataset.clone(),
            fuzzy_query.clone(),
            search_params.clone(),
            PreFilterSource::None,
            vec![preset_segments[0].clone()],
        )
        .unwrap()
        .with_base_scorer(prepared.scorer().clone());
        assert_execution_error(
            execute_row_ids(&scorer_only_exec).await.unwrap_err(),
            "fuzzy MatchQuery cannot use a scorer-only override",
        );

        let compound_query = FtsQuery::Match(fuzzy_query.clone());
        let compound_subset = CompoundQueryExec::new_with_segments(
            dataset.clone(),
            compound_query.clone(),
            search_params.clone(),
            PreFilterSource::None,
            vec![preset_segments[0].clone()],
        )
        .with_prepared_match(prepared.clone());
        assert!(execute_row_ids(&compound_subset).await.unwrap().is_empty());

        let compound_scorer_only_subset = CompoundQueryExec::new_with_segments(
            dataset.clone(),
            compound_query.clone(),
            search_params.clone(),
            PreFilterSource::None,
            vec![preset_segments[0].clone()],
        )
        .with_base_scorer(prepared.scorer().clone());
        assert_execution_error(
            execute_row_ids(&compound_scorer_only_subset)
                .await
                .unwrap_err(),
            "fuzzy CompoundQueryExec cannot use a scorer-only override over a segment subset",
        );

        let compound_full_scorer = CompoundQueryExec::new_with_segments(
            dataset.clone(),
            compound_query.clone(),
            search_params.clone(),
            PreFilterSource::None,
            preset_segments.clone(),
        )
        .with_base_scorer(prepared.scorer().clone());
        assert_eq!(
            execute_row_ids(&compound_full_scorer).await.unwrap(),
            prepared_full_results
        );

        // With limit=1 the two globally selected `lancd` documents tie. The
        // WAND probe is ambiguous, so bounded tie completion must retain the
        // same prepared vocabulary instead of rewriting against this exec's
        // segments.
        let compound_wand_replay = CompoundQueryExec::new_with_segments(
            dataset,
            compound_query,
            search_params.with_limit(Some(1)),
            PreFilterSource::None,
            preset_segments,
        )
        .with_prepared_match(prepared);
        assert_eq!(
            execute_row_ids(&compound_wand_replay).await.unwrap().len(),
            1
        );
        // Locally-bound helper: collect (row_id, score) pairs sorted by score desc.
        fn concat_score_batches(batches: &[RecordBatch]) -> Vec<(u64, f32)> {
            let mut out: Vec<(u64, f32)> = Vec::new();
            for batch in batches {
                let row_ids = batch
                    .column_by_name(ROW_ID)
                    .unwrap()
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap();
                let scores = batch
                    .column_by_name(SCORE_COL)
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .unwrap();
                for i in 0..batch.num_rows() {
                    out.push((row_ids.value(i), scores.value(i)));
                }
            }
            // Stable order for diffing — descending score, ties broken by row id.
            out.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            out
        }
    }

    #[tokio::test]
    async fn test_compound_query_exec_validates_base_scorer() {
        let (dataset, segments, _) = create_segment_selection_fixture().await;
        let search_params = FtsSearchParams::default().with_limit(Some(10));
        let metrics_set = ExecutionPlanMetricsSet::new();
        let metrics = IndexMetrics::new(&metrics_set, 0);
        let indices = open_fts_segments(&dataset, "text", &segments, &metrics)
            .await
            .unwrap();

        let query: FtsQuery = BooleanQuery::new([
            (
                Occur::Should,
                MatchQuery::new("quick".to_string())
                    .with_column(Some("text".to_string()))
                    .into(),
            ),
            (
                Occur::Should,
                MatchQuery::new("brown".to_string())
                    .with_column(Some("text".to_string()))
                    .into(),
            ),
        ])
        .into();

        let baseline = CompoundQueryExec::new_with_segments(
            dataset.clone(),
            query.clone(),
            search_params.clone(),
            PreFilterSource::None,
            segments.clone(),
        );
        let baseline_results = execute_results(&baseline).await.unwrap();

        let mut tokenizer = indices[0].tokenizer();
        let complete_tokens = try_collect_query_tokens("quick brown", &mut tokenizer).unwrap();
        let complete_scorer = Arc::new(
            build_global_bm25_scorer(&indices, &complete_tokens, &search_params, None)
                .await
                .unwrap(),
        );
        let complete_override = CompoundQueryExec::new_with_segments(
            dataset.clone(),
            query.clone(),
            search_params.clone(),
            PreFilterSource::None,
            segments.clone(),
        )
        .with_base_scorer(complete_scorer);
        assert_eq!(
            execute_results(&complete_override).await.unwrap(),
            baseline_results
        );

        let mut tokenizer = indices[0].tokenizer();
        let incomplete_tokens = try_collect_query_tokens("quick", &mut tokenizer).unwrap();
        let incomplete_scorer = Arc::new(
            build_global_bm25_scorer(&indices, &incomplete_tokens, &search_params, None)
                .await
                .unwrap(),
        );
        let incomplete_override = CompoundQueryExec::new_with_segments(
            dataset.clone(),
            query,
            search_params.clone(),
            PreFilterSource::None,
            segments.clone(),
        )
        .with_base_scorer(incomplete_scorer);

        let error = execute_results(&incomplete_override).await.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("injected BM25 scorer is missing compound FTS token 'brown'"),
            "unexpected incomplete-scorer error: {error}"
        );

        let mut tokenizer = indices[0].tokenizer();
        let brown_tokens = try_collect_query_tokens("brown", &mut tokenizer).unwrap();
        let scorer_without_fuzzy_expansion = Arc::new(
            build_global_bm25_scorer(&indices, &brown_tokens, &search_params, None)
                .await
                .unwrap(),
        );
        let fuzzy_query = BooleanQuery::new([
            (
                Occur::Should,
                MatchQuery::new("quik".to_string())
                    .with_column(Some("text".to_string()))
                    .with_fuzziness(Some(1))
                    .into(),
            ),
            (
                Occur::Should,
                MatchQuery::new("brown".to_string())
                    .with_column(Some("text".to_string()))
                    .into(),
            ),
        ]);
        let fuzzy_override = CompoundQueryExec::new_with_segments(
            dataset,
            fuzzy_query.into(),
            search_params,
            PreFilterSource::None,
            segments,
        )
        .with_base_scorer(scorer_without_fuzzy_expansion);
        let error = execute_results(&fuzzy_override).await.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("injected BM25 scorer is missing compound FTS token 'quick'"),
            "unexpected fuzzy-scorer error: {error}"
        );
    }

    #[tokio::test]
    async fn test_cross_column_compound_exec_validates_constructor_inputs() {
        let (dataset, segments, _) = create_segment_selection_fixture().await;
        let query: FtsQuery = BooleanQuery::new([
            (
                Occur::Must,
                MatchQuery::new("quick".to_string())
                    .with_column(Some("title".to_string()))
                    .into(),
            ),
            (
                Occur::MustNot,
                MatchQuery::new("blocked".to_string())
                    .with_column(Some("body".to_string()))
                    .into(),
            ),
        ])
        .into();
        let params = FtsSearchParams::default().with_limit(Some(10));

        let error = CrossColumnCompoundQueryExec::new_with_segments(
            dataset.clone(),
            query.clone(),
            params.clone(),
            PreFilterSource::None,
            vec![("title".to_string(), segments.clone())],
        )
        .unwrap_err();
        assert!(
            error.to_string().contains(r#"missing=["body"]"#),
            "unexpected missing-column error: {error}"
        );

        let error = CrossColumnCompoundQueryExec::new_with_segments(
            dataset.clone(),
            query.clone(),
            FtsSearchParams::default(),
            PreFilterSource::None,
            vec![
                ("title".to_string(), segments.clone()),
                ("body".to_string(), segments.clone()),
            ],
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("requires a bounded result limit"),
            "unexpected unbounded-query error: {error}"
        );

        let exec = CrossColumnCompoundQueryExec::new_with_segments(
            dataset,
            query,
            params,
            PreFilterSource::None,
            vec![
                ("title".to_string(), segments.clone()),
                ("body".to_string(), segments),
            ],
        )
        .unwrap();
        let display = format!(
            "{}",
            datafusion::physical_plan::displayable(&exec).one_line()
        );
        assert!(
            display.contains("CrossColumnCompoundFtsScorer:"),
            "unexpected display name: {display}"
        );
    }

    fn empty_fts_child() -> Arc<dyn ExecutionPlan> {
        Arc::new(EmptyExec::new(FTS_SCHEMA.clone()))
    }

    #[test]
    fn build_boolean_should_empty_returns_empty_exec() {
        let plan = build_boolean_query_children(BoolSlot::Should, vec![])
            .unwrap()
            .expect("Should slot always returns Some");
        assert!(
            plan.downcast_ref::<EmptyExec>().is_some(),
            "expected EmptyExec for empty Should slot, got {plan:?}"
        );
    }

    #[test]
    fn build_boolean_should_single_child_passthrough() {
        let child = empty_fts_child();
        let child_ptr = Arc::as_ptr(&child);
        let plan = build_boolean_query_children(BoolSlot::Should, vec![child])
            .unwrap()
            .expect("Should slot always returns Some");
        assert_eq!(
            Arc::as_ptr(&plan),
            child_ptr,
            "single-child Should should return the child unchanged"
        );
    }

    #[test]
    fn build_boolean_should_multi_child_union_repartition() {
        let plan = build_boolean_query_children(
            BoolSlot::Should,
            vec![empty_fts_child(), empty_fts_child()],
        )
        .unwrap()
        .expect("Should slot always returns Some");
        let repartition = plan
            .downcast_ref::<RepartitionExec>()
            .expect("multi-child Should should be wrapped in RepartitionExec");
        let inner = repartition
            .input()
            .downcast_ref::<UnionExec>()
            .expect("RepartitionExec should wrap a UnionExec");
        assert_eq!(inner.children().len(), 2);
    }

    #[test]
    fn build_boolean_must_empty_returns_none() {
        let plan = build_boolean_query_children(BoolSlot::Must, vec![]).unwrap();
        assert!(plan.is_none(), "empty Must slot should return None");
    }

    #[test]
    fn build_boolean_must_single_child_passthrough_some() {
        let child = empty_fts_child();
        let child_ptr = Arc::as_ptr(&child);
        let plan = build_boolean_query_children(BoolSlot::Must, vec![child])
            .unwrap()
            .expect("single-child Must should be Some");
        assert_eq!(
            Arc::as_ptr(&plan),
            child_ptr,
            "single-child Must should return the child unchanged"
        );
    }

    #[test]
    fn build_boolean_must_multi_child_chained_hashjoin() {
        let children = vec![empty_fts_child(), empty_fts_child(), empty_fts_child()];
        let n = children.len();
        let plan = build_boolean_query_children(BoolSlot::Must, children)
            .unwrap()
            .expect("multi-child Must should be Some");

        // Walk the left spine: each layer is a HashJoinExec whose left child is
        // either another HashJoinExec or the original leaf. With N children
        // there are N-1 joins.
        let mut joins = 0usize;
        let mut current: Arc<dyn ExecutionPlan> = plan;
        while let Some(join) = current.clone().downcast_ref::<HashJoinExec>() {
            joins += 1;
            current = join.children()[0].clone();
        }
        assert_eq!(joins, n - 1, "expected {} joins for {n} children", n - 1);
    }

    #[test]
    fn build_boolean_must_not_multi_child_union_repartition() {
        let plan = build_boolean_query_children(
            BoolSlot::MustNot,
            vec![empty_fts_child(), empty_fts_child()],
        )
        .unwrap()
        .expect("MustNot slot always returns Some");
        let repartition = plan
            .downcast_ref::<RepartitionExec>()
            .expect("multi-child MustNot should be wrapped in RepartitionExec");
        let inner = repartition
            .input()
            .downcast_ref::<UnionExec>()
            .expect("RepartitionExec should wrap a UnionExec");
        assert_eq!(inner.children().len(), 2);
    }

    /// `(id, score)` of every hit, sorted by id. The limit lets a query over several
    /// columns use the cross-column scorer, which is skipped without one.
    async fn fts_hits(dataset: &Dataset, query: FtsQuery) -> Vec<(String, f32)> {
        let mut scan = dataset.scan();
        scan.full_text_search(FullTextSearchQuery::new_query(query).limit(Some(10)))
            .unwrap();
        scan.project(&["id"]).unwrap();
        let batch = scan.try_into_batch().await.unwrap();
        let ids = batch["id"].as_any().downcast_ref::<StringArray>().unwrap();
        let scores = batch[SCORE_COL]
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        let mut hits = ids
            .iter()
            .zip(scores.values())
            .map(|(id, score)| (id.unwrap().to_string(), *score))
            .collect::<Vec<_>>();
        hits.sort_by(|a, b| a.0.cmp(&b.0));
        hits
    }

    /// Every FTS search path reports only current rows: the old segment still lists `a1`
    /// under "graph engines", but `a1` now lives in a fragment the newer segment covers.
    async fn assert_no_stale_rows(dataset: &Dataset) {
        let ids = |hits: Vec<(String, f32)>| hits.into_iter().map(|(id, _)| id).collect::<Vec<_>>();
        let text = |terms: &str| -> FtsQuery {
            MatchQuery::new(terms.into())
                .with_column(Some("text".into()))
                .into()
        };
        assert_eq!(
            ids(fts_hits(dataset, text("graph")).await),
            ["a2", "a3", "a5"]
        );
        // Listed once, not once per segment.
        assert_eq!(ids(fts_hits(dataset, text("engines")).await), ["a1", "a4"]);
        let phrase = PhraseQuery::new("graph engines".into()).with_column(Some("text".into()));
        assert_eq!(fts_hits(dataset, phrase.into()).await, []);
        // `tag:zeta` (3 postings, stale `a1` included) is cheaper than `text:graph` (4),
        // so it generates the candidates and `text:graph` is only loaded for them.
        let tag_zeta = MatchQuery::new("zeta".into()).with_column(Some("tag".into()));
        let cross =
            BooleanQuery::new([(Occur::Must, tag_zeta.into()), (Occur::Must, text("graph"))]);
        assert_eq!(fts_hits(dataset, FtsQuery::Boolean(cross)).await, []);
        // `a1` and `a4` hold identical text, so `a1`'s old length must not count.
        let combined =
            CombinedFieldsQuery::try_new("zeta".into(), vec!["text".into(), "tag".into()]).unwrap();
        let hits = fts_hits(dataset, FtsQuery::CombinedFields(combined)).await;
        assert_eq!(ids(hits.clone()), ["a1", "a4"]);
        assert_eq!(hits[0].1, hits[1].1);
    }

    /// https://github.com/lance-format/lance/issues/9817: under stable row ids an
    /// in-place update moves `a1` to a new fragment with the same row id, and an
    /// append-only optimize gives that fragment its own segment. Merging the segments
    /// must drop the stale posting rather than carry it into one segment.
    #[rstest::rstest]
    #[case::optimize_indices(true)]
    #[case::merge_existing_index_segments(false)]
    #[tokio::test]
    async fn stale_posting_after_in_place_update_with_stable_row_ids(
        #[case] merge_by_optimize: bool,
    ) {
        let base = record_batch!(
            ("id", Utf8, ["a1", "a2", "a3", "a4", "a5"]),
            (
                "text",
                Utf8,
                [
                    "graph engines",
                    "graph search",
                    "graph databases",
                    "omega engines",
                    "graph theory"
                ]
            ),
            ("tag", Utf8, ["zeta", "misc", "misc", "zeta", "misc"])
        )
        .unwrap();
        let schema = base.schema();
        let params = WriteParams {
            enable_stable_row_ids: true,
            ..Default::default()
        };
        let reader = RecordBatchIterator::new(vec![Ok(base)], schema.clone());
        let mut dataset = Dataset::write(reader, "memory://", Some(params))
            .await
            .unwrap();
        let index_params = InvertedIndexParams::default().with_position(true);
        for column in ["text", "tag"] {
            dataset
                .create_index(&[column], IndexType::Inverted, None, &index_params, true)
                .await
                .unwrap();
        }
        let engines = MatchQuery::new("engines".into()).with_column(Some("text".into()));
        assert_eq!(
            fts_hits(&dataset, engines.into())
                .await
                .into_iter()
                .map(|(id, _)| id)
                .collect::<Vec<_>>(),
            ["a1", "a4"]
        );

        let update = record_batch!(
            ("id", Utf8, ["a1"]),
            ("text", Utf8, ["omega engines"]),
            ("tag", Utf8, ["zeta"])
        )
        .unwrap();
        let mut merge = MergeInsertBuilder::try_new(Arc::new(dataset), vec!["id".into()]).unwrap();
        merge
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::DoNothing);
        let reader = RecordBatchIterator::new(vec![Ok(update)], schema);
        let (dataset, _) = merge
            .try_build()
            .unwrap()
            .execute_reader(reader)
            .await
            .unwrap();
        let mut dataset = dataset.as_ref().clone();
        dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .unwrap();
        let segments = dataset.load_indices_by_name("text_idx").await.unwrap();
        assert_eq!(segments.len(), 2);
        assert_no_stale_rows(&dataset).await;

        if merge_by_optimize {
            let options = OptimizeOptions::default().num_indices_to_merge(Some(2));
            dataset.optimize_indices(&options).await.unwrap();
        } else {
            let merged = dataset
                .merge_existing_index_segments(segments)
                .await
                .unwrap();
            dataset
                .commit_existing_index_segments("text_idx", "text", vec![merged])
                .await
                .unwrap();
        }
        let segments = dataset.load_indices_by_name("text_idx").await.unwrap();
        assert_eq!(segments.len(), 1);
        assert_no_stale_rows(&dataset).await;
    }

    /// `num_segments` appends, each its own FTS segment, with a `userid`
    /// BTree so a `userid` range is an index-only prefilter whose rows lie
    /// in a known subset of the segments.
    async fn fts_segment_fixture(
        num_segments: usize,
        rows_per_segment: usize,
        stable_row_ids: bool,
    ) -> (Dataset, TempStrDir) {
        let make = |start: u64| {
            lance_datagen::gen_batch()
                .col(
                    "text",
                    lance_datagen::array::cycle_utf8_literals(&[
                        "hello world",
                        "lance search",
                        "hello lance",
                    ]),
                )
                .col(
                    "text2",
                    lance_datagen::array::cycle_utf8_literals(&[
                        "hello again",
                        "other words",
                        "hello there",
                        "nothing",
                    ]),
                )
                .col(
                    "userid",
                    lance_datagen::array::step_custom::<arrow::datatypes::UInt64Type>(start, 1),
                )
                .into_reader_rows(RowCount::from(rows_per_segment as u64), BatchCount::from(1))
        };
        let dir = TempStrDir::default();
        let mut dataset = Dataset::write(
            make(0),
            dir.as_str(),
            Some(WriteParams {
                enable_stable_row_ids: stable_row_ids,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        for column in ["text", "text2"] {
            dataset
                .create_index(
                    &[column],
                    IndexType::Inverted,
                    None,
                    &InvertedIndexParams::default().with_position(true),
                    true,
                )
                .await
                .unwrap();
        }
        for segment in 1..num_segments {
            dataset
                .append(make((segment * rows_per_segment) as u64), None)
                .await
                .unwrap();
            dataset
                .optimize_indices(&OptimizeOptions::append())
                .await
                .unwrap();
        }
        dataset
            .create_index(
                &["userid"],
                IndexType::BTree,
                None,
                &lance_index::scalar::ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();
        assert_eq!(
            dataset
                .load_indices_by_name("text_idx")
                .await
                .unwrap()
                .len(),
            num_segments
        );
        (dataset, dir)
    }

    /// Every hit of `query` under `filter` as (row id, score), by row id.
    async fn fts_rows(
        dataset: &Dataset,
        query: FullTextSearchQuery,
        filter: &str,
        prefilter: bool,
        stats: Option<&StatsHolder>,
    ) -> Vec<(u64, f32)> {
        let mut scanner = dataset.scan();
        scanner
            .project(&["userid"])
            .unwrap()
            .with_row_id()
            .full_text_search(query.limit(Some(10_000)))
            .unwrap()
            .filter(filter)
            .unwrap()
            .prefilter(prefilter);
        if let Some(stats) = stats {
            scanner.scan_stats_callback(stats.get_setter());
        }
        let batch = scanner.try_into_batch().await.unwrap();
        let ids = batch.column_by_name(ROW_ID).unwrap();
        let ids = ids.as_primitive::<UInt64Type>();
        let scores = batch.column_by_name(SCORE_COL).unwrap();
        let scores = scores.as_primitive::<Float32Type>();
        let mut rows: Vec<(u64, f32)> = (0..batch.num_rows())
            .map(|i| (ids.value(i), scores.value(i)))
            .collect();
        rows.sort_by_key(|row| row.0);
        rows
    }

    /// An index-only prefilter whose rows lie in some segments only lets the
    /// candidate search of the others be skipped; the corpus statistics and
    /// therefore the scores are those of the unpruned search, and a block
    /// list proves nothing.
    #[rstest]
    #[tokio::test]
    async fn prefilter_skips_fts_segments_it_excludes(#[values(false, true)] stable_row_ids: bool) {
        use lance_index::scalar::inverted::query::{BooleanQuery, Occur};
        let (dataset, _dir) = fts_segment_fixture(3, 30, stable_row_ids).await;
        let text =
            |terms: &str| MatchQuery::new(terms.to_string()).with_column(Some("text".to_string()));
        // Match and Phrase prune their candidate search; a Boolean query from
        // the scanner derives its statistics from the segments it searches and
        // therefore keeps them all (the prepared arms are covered separately).
        let queries: Vec<(&str, FullTextSearchQuery, bool)> = vec![
            // Explicit column: with two indexed text columns a bare term would
            // become a multi-column match.
            (
                "match",
                FullTextSearchQuery::new_query(FtsQuery::Match(text("hello"))),
                true,
            ),
            (
                "phrase",
                FullTextSearchQuery::new_query(FtsQuery::Phrase(
                    PhraseQuery::new("hello world".to_string())
                        .with_column(Some("text".to_string())),
                )),
                true,
            ),
            (
                "boolean",
                FullTextSearchQuery::new_query(FtsQuery::Boolean(BooleanQuery::new([(
                    Occur::Must,
                    FtsQuery::Match(text("hello")),
                )]))),
                false,
            ),
            // Two target columns, each with its own segments: every excluded
            // segment of either column is skipped, so the count doubles.
            (
                "combined_fields",
                FullTextSearchQuery::new_query(FtsQuery::CombinedFields(
                    lance_index::scalar::inverted::query::CombinedFieldsQuery::try_new(
                        "hello".to_string(),
                        vec!["text".to_string(), "text2".to_string()],
                    )
                    .unwrap(),
                )),
                true,
            ),
        ];
        for (kind, query, prunes) in &queries {
            for (filter, excluded) in [
                ("userid < 30", 2),
                ("userid >= 60", 2),
                ("userid >= 30", 1),
                ("userid < 1000", 0),
                ("userid > 1000", 3),
                ("userid != 5", 0),
            ] {
                let stats = StatsHolder::default();
                let pruned = fts_rows(&dataset, query.clone(), filter, true, Some(&stats)).await;
                let oracle = fts_rows(&dataset, query.clone(), filter, false, None).await;
                assert_eq!(pruned, oracle, "{kind}: {filter}");
                if filter == "userid < 1000" {
                    assert!(!pruned.is_empty(), "{kind}: the fixture has hits");
                }
                let counts = stats.consume().all_counts;
                let columns = if *kind == "combined_fields" { 2 } else { 1 };
                assert_eq!(
                    counts
                        .get(INDEX_SEGMENTS_SKIPPED_METRIC)
                        .copied()
                        .unwrap_or_default(),
                    if *prunes { excluded * columns } else { 0 },
                    "{kind}: {filter}"
                );
            }
        }
    }

    /// The Compound prepared arms, driven directly: a complete prefilter that
    /// excludes every segment yields an empty result rather than an error,
    /// a partial one yields the oracle's rows, through the WAND certificate
    /// path (with a limit) and the plain prepared-match arm (without); the
    /// base-scorer arm stays unpruned and agrees.
    #[tokio::test]
    async fn compound_prepared_arms_handle_empty_and_partial_candidate_domains() {
        use crate::io::exec::scalar_index::ScalarIndexExec;
        use lance_index::scalar::SargableQuery;
        use lance_index::scalar::expression::{ScalarIndexExpr, ScalarIndexSearch};
        use std::ops::Bound;

        let (dataset, _dir) = fts_segment_fixture(3, 30, false).await;
        let dataset = Arc::new(dataset);
        let segments = dataset.load_indices_by_name("text_idx").await.unwrap();
        let metrics_set = ExecutionPlanMetricsSet::new();
        let metrics = IndexMetrics::new(&metrics_set, 0);
        let indices = open_fts_segments(&dataset, "text", &segments, &metrics)
            .await
            .unwrap();
        let query = MatchQuery::new("hello".to_string()).with_column(Some("text".to_string()));
        let mut tokenizer = tokenizer_for_match_query(&indices[0], query.fuzziness);
        let tokens = collect_query_tokens(&query.terms, &mut tokenizer);
        let userid = |query: SargableQuery| {
            ScalarIndexExpr::Query(ScalarIndexSearch {
                column: "userid".to_string(),
                index_name: "userid_idx".to_string(),
                index_type: "BTree".to_string(),
                query: Arc::new(query),
                needs_recheck: false,
                fragment_bitmap: None,
            })
        };
        let u = |v: u64| datafusion::scalar::ScalarValue::UInt64(Some(v));
        let cases = [
            (
                userid(SargableQuery::Range(
                    Bound::Excluded(u(1000)),
                    Bound::Unbounded,
                )),
                "userid > 1000",
            ),
            (
                userid(SargableQuery::Range(
                    Bound::Unbounded,
                    Bound::Excluded(u(30)),
                )),
                "userid < 30",
            ),
            (
                userid(SargableQuery::Range(
                    Bound::Unbounded,
                    Bound::Excluded(u(1000)),
                )),
                "userid < 1000",
            ),
        ];
        for (expr, filter) in cases {
            let oracle = dataset
                .scan()
                .filter(&format!("{filter} AND text LIKE '%hello%'"))
                .unwrap()
                .with_row_id()
                .project(&Vec::<String>::new())
                .unwrap()
                .try_into_batch()
                .await
                .unwrap();
            let mut expected: Vec<u64> = oracle
                .column_by_name(ROW_ID)
                .unwrap()
                .as_primitive::<UInt64Type>()
                .values()
                .to_vec();
            expected.sort_unstable();
            for limit in [Some(1000), None] {
                let params = FtsSearchParams::default().with_limit(limit);
                let prepared = Arc::new(
                    prepare_bm25_query(&indices, tokens.clone(), &params, None, None)
                        .await
                        .unwrap(),
                );
                let prefilter = || {
                    PreFilterSource::ScalarIndexQuery(Arc::new(ScalarIndexExec::new(
                        dataset.clone(),
                        expr.clone(),
                        IndexExprResultWireFormat::default(),
                    ))
                        as Arc<dyn ExecutionPlan>)
                };
                let exec = |prepared_match: bool| {
                    let exec = CompoundQueryExec::new_with_segments(
                        dataset.clone(),
                        FtsQuery::Match(query.clone()),
                        params.clone(),
                        prefilter(),
                        segments.clone(),
                    );
                    if prepared_match {
                        exec.with_prepared_match(prepared.clone())
                    } else {
                        exec.with_base_scorer(prepared.scorer().clone())
                    }
                };
                for (arm, prepared_match) in [("prepared match", true), ("base scorer", false)] {
                    let mut got = execute_row_ids(&exec(prepared_match)).await.unwrap();
                    got.sort_unstable();
                    assert_eq!(got, expected, "{arm}, limit {limit:?}, {filter}");
                }
            }
        }
    }
}
