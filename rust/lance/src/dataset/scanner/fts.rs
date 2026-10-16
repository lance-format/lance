// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_schema::{Schema as ArrowSchema, SortOptions};
use async_recursion::async_recursion;
use datafusion::common::{JoinType, NullEquality};
use datafusion::functions_aggregate;
use datafusion::physical_expr::PhysicalSortExpr;
use datafusion::physical_plan::aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy};
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::union::UnionExec;
use datafusion::physical_plan::{ExecutionPlan, expressions};
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::{Partitioning, PhysicalExpr};
use datafusion_physical_plan::empty::EmptyExec;
use datafusion_physical_plan::joins::{HashJoinExec, PartitionMode};
use datafusion_physical_plan::projection::ProjectionExec;
use lance_arrow::SchemaExt as ArrowSchemaExt;
use lance_core::ROW_ID;
use lance_core::datatypes::OnMissing;
use lance_index::IndexCriteria;
use lance_index::pbold::InvertedIndexDetails;
use lance_index::scalar::FullTextSearchQuery;
use lance_index::scalar::inverted::query::{
    CombinedFieldsQuery, FtsQuery, FtsSearchParams, MatchQuery, Operator, PhraseQuery,
    fill_fts_query_column,
};
use lance_index::scalar::inverted::{
    CombinedFieldsBM25Scorer, DocumentGranularity, INVERTED_INDEX_VERSION_V2,
    INVERTED_INDEX_VERSION_V3, SCORE_COL, SCORE_FIELD, fts_schema,
};
use lance_index::scalar::registry::VALUE_COLUMN_NAME;
use lance_select::RowAddrTreeMap;
use lance_table::format::{Fragment, IndexMetadata};
use prost::Message;
use roaring::RoaringBitmap;

use crate::dataset::Dataset;
use crate::dataset::overlay::{collect_overlay_stale_rows_for_segment, overlaid_fragments};
use crate::index::scalar::fetch_index_details;
use crate::index::scalar::inverted::{
    fts_index_fragment_bitmap, load_segment_details, load_segments, normalize_inverted_details,
    resolve_fts_field, resolve_query_document_granularity, validate_combined_fields_target_column,
};
use crate::index::{DatasetIndexExt, DatasetIndexInternalExt};
use crate::io::exec::fts::{
    BoolSlot, BooleanQueryExec, BoostQueryExec, CombinedFieldsQueryExec, CompoundQueryExec,
    CrossColumnCompoundQueryExec, FlatCombinedFieldsExec, FlatMatchFilterExec, FlatMatchQueryExec,
    FlatStatsCoverage, FlatStatsScope, FtsDocumentExec, HybridCompoundQueryExec, MatchQueryExec,
    PhraseQueryExec, SharedFtsScorer, build_boolean_query_children_with_schema,
};
use crate::io::exec::{FilterPlan as ExprFilterPlan, PreFilterSource, RowAddrMaskFilterExec};
use crate::{Error, Result};

use super::{FilterPlan, PlannedFilteredScan, Scanner, fts_indexed_columns};

enum FtsOverlayPlan {
    Unchanged(Option<Vec<IndexMetadata>>),
    RowLevel {
        stale_rows: HashMap<u32, RoaringBitmap>,
        segments: Vec<IndexMetadata>,
    },
    FullScan,
}

fn collect_fts_columns_in_order(query: &FtsQuery) -> Vec<String> {
    fn visit(query: &FtsQuery, columns: &mut Vec<String>, seen: &mut HashSet<String>) {
        match query {
            FtsQuery::Match(query) => {
                if let Some(column) = &query.column
                    && seen.insert(column.clone())
                {
                    columns.push(column.clone());
                }
            }
            FtsQuery::Phrase(query) => {
                if let Some(column) = &query.column
                    && seen.insert(column.clone())
                {
                    columns.push(column.clone());
                }
            }
            FtsQuery::Boost(query) => {
                visit(&query.positive, columns, seen);
                visit(&query.negative, columns, seen);
            }
            FtsQuery::MultiMatch(query) => {
                for match_query in &query.match_queries {
                    if let Some(column) = &match_query.column
                        && seen.insert(column.clone())
                    {
                        columns.push(column.clone());
                    }
                }
            }
            FtsQuery::Boolean(query) => {
                for child in query
                    .should
                    .iter()
                    .chain(&query.must)
                    .chain(&query.must_not)
                {
                    visit(child, columns, seen);
                }
            }
            FtsQuery::CombinedFields(query) => {
                for column in query.column_names() {
                    if seen.insert(column.to_string()) {
                        columns.push(column.to_string());
                    }
                }
            }
        }
    }

    let mut columns = Vec::new();
    let mut seen = HashSet::new();
    visit(query, &mut columns, &mut seen);
    columns
}

fn collect_phrase_columns(query: &FtsQuery, columns: &mut HashSet<String>) {
    match query {
        FtsQuery::Phrase(query) => {
            if let Some(column) = &query.column {
                columns.insert(column.clone());
            }
        }
        FtsQuery::Boost(query) => {
            collect_phrase_columns(&query.positive, columns);
            collect_phrase_columns(&query.negative, columns);
        }
        FtsQuery::Boolean(query) => {
            for child in query
                .should
                .iter()
                .chain(&query.must)
                .chain(&query.must_not)
            {
                collect_phrase_columns(child, columns);
            }
        }
        // combined_fields is term-based; it never needs positions.
        FtsQuery::Match(_) | FtsQuery::MultiMatch(_) | FtsQuery::CombinedFields(_) => {}
    }
}

async fn load_physical_fts_details(
    dataset: &Dataset,
    column: &str,
    segment: &IndexMetadata,
) -> Result<InvertedIndexDetails> {
    let details = fetch_index_details(dataset, column, segment).await?;
    let details = InvertedIndexDetails::decode(details.value.as_slice()).map_err(|err| {
        Error::io(format!(
            "failed to decode InvertedIndexDetails payload: {err}"
        ))
    })?;
    normalize_inverted_details(segment, details)
}

fn supports_compound_scorer(query: &FtsQuery) -> bool {
    fn supports_shape(query: &FtsQuery) -> bool {
        match query {
            FtsQuery::Match(_) | FtsQuery::Phrase(_) | FtsQuery::MultiMatch(_) => true,
            // BM25F blends term statistics across columns, so it cannot be a leaf
            // of the single-index compound scorer. Such trees keep the
            // union/sort plan built by `plan_combined_fields_query`.
            FtsQuery::CombinedFields(_) => false,
            FtsQuery::Boolean(query) => {
                (!query.should.is_empty() || !query.must.is_empty())
                    && query
                        .should
                        .iter()
                        .chain(&query.must)
                        .chain(&query.must_not)
                        .all(supports_shape)
            }
            FtsQuery::Boost(query) => {
                supports_shape(&query.positive) && supports_shape(&query.negative)
            }
        }
    }

    if matches!(query, FtsQuery::Match(_) | FtsQuery::Phrase(_)) || !supports_shape(query) {
        return false;
    }
    let columns = collect_fts_columns_in_order(query);
    !columns.is_empty() && (!matches!(query, FtsQuery::MultiMatch(_)) || columns.len() == 1)
}

fn supports_indexed_stats_residual_compound(query: &FtsQuery) -> bool {
    match query {
        FtsQuery::Match(query) => query.fuzziness == Some(0),
        // MemWAL phrase matching currently collapses tokenizer position gaps.
        // Keep phrase queries on the established fallback until it can retain
        // those gaps exactly (notably when stop words are configured).
        FtsQuery::Phrase(_) => false,
        FtsQuery::Boost(query) => {
            supports_indexed_stats_residual_compound(&query.positive)
                && supports_indexed_stats_residual_compound(&query.negative)
        }
        FtsQuery::MultiMatch(query) => query
            .match_queries
            .iter()
            .all(|query| query.fuzziness == Some(0)),
        FtsQuery::Boolean(query) => query
            .should
            .iter()
            .chain(&query.must)
            .chain(&query.must_not)
            .all(supports_indexed_stats_residual_compound),
        // The compound scorer scores each leaf from one column's index, which a
        // cross-field blend cannot be expressed as.
        FtsQuery::CombinedFields(_) => false,
    }
}

const MAX_QUERY_LOCAL_RESIDUAL_ROWS: usize = 100_000;

fn has_bounded_query_local_residual_rows(fragments: &[Fragment]) -> bool {
    fragments
        .iter()
        .try_fold(0usize, |total, fragment| {
            total.checked_add(fragment.physical_rows?)
        })
        .is_some_and(|total| total <= MAX_QUERY_LOCAL_RESIDUAL_ROWS)
}

fn has_complete_hybrid_fts_coverage(
    segments: &[IndexMetadata],
    residual_fragments: &[Fragment],
    target_fragments: &[Fragment],
) -> bool {
    let Some(target) = target_fragments
        .iter()
        .map(|fragment| u32::try_from(fragment.id).ok())
        .collect::<Option<RoaringBitmap>>()
    else {
        return false;
    };
    let Some(residual) = residual_fragments
        .iter()
        .map(|fragment| u32::try_from(fragment.id).ok())
        .collect::<Option<RoaringBitmap>>()
    else {
        return false;
    };
    let mut indexed = RoaringBitmap::new();
    for segment in segments {
        let Some(coverage) = segment.fragment_bitmap.as_ref() else {
            return false;
        };
        if !indexed.is_disjoint(coverage) {
            return false;
        }
        indexed |= coverage;
    }
    if !indexed.is_subset(&target) || !indexed.is_disjoint(&residual) {
        return false;
    }
    indexed | residual == target
}

fn validate_fts_query_contract(query: &FtsQuery) -> Result<()> {
    fn validate_multiplier(name: &str, value: f32) -> Result<()> {
        if value.is_finite() && value >= 0.0 {
            Ok(())
        } else {
            Err(Error::invalid_input(format!(
                "{name} must be finite and non-negative, got {value}"
            )))
        }
    }

    match query {
        FtsQuery::Match(query) => validate_multiplier("MatchQuery boost", query.boost),
        FtsQuery::Phrase(_) => Ok(()),
        // Nothing to re-check: the weights are private and every constructor,
        // including `Deserialize`, goes through `try_with_boosts`, which enforces a
        // finite value in `[MIN_BOOST, MAX_BOOST]`, stricter than the non-negative
        // contract checked here.
        FtsQuery::CombinedFields(_) => Ok(()),
        FtsQuery::Boost(query) => {
            validate_multiplier("BoostQuery negative_boost", query.negative_boost)?;
            validate_fts_query_contract(&query.positive)?;
            validate_fts_query_contract(&query.negative)
        }
        FtsQuery::MultiMatch(query) => {
            for match_query in &query.match_queries {
                validate_multiplier("MultiMatchQuery boost", match_query.boost)?;
            }
            Ok(())
        }
        FtsQuery::Boolean(query) => {
            if query.should.is_empty() && query.must.is_empty() {
                return Err(Error::invalid_input(
                    "boolean query must have at least one should/must query",
                ));
            }
            for child in query
                .should
                .iter()
                .chain(&query.must)
                .chain(&query.must_not)
            {
                validate_fts_query_contract(child)?;
            }
            Ok(())
        }
    }
}

fn normalize_fts_zero_boosts(query: &mut FtsQuery) {
    fn normalize_zero(value: &mut f32) {
        if *value == 0.0 {
            *value = 0.0;
        }
    }

    match query {
        FtsQuery::Match(query) => normalize_zero(&mut query.boost),
        // combined_fields weights are validated into `[1.0, 2^20]`, so no zero
        // to normalize.
        FtsQuery::Phrase(_) | FtsQuery::CombinedFields(_) => {}
        FtsQuery::Boost(query) => {
            normalize_zero(&mut query.negative_boost);
            normalize_fts_zero_boosts(&mut query.positive);
            normalize_fts_zero_boosts(&mut query.negative);
        }
        FtsQuery::MultiMatch(query) => {
            for match_query in &mut query.match_queries {
                normalize_zero(&mut match_query.boost);
            }
        }
        FtsQuery::Boolean(query) => {
            for child in query
                .should
                .iter_mut()
                .chain(&mut query.must)
                .chain(&mut query.must_not)
            {
                normalize_fts_zero_boosts(child);
            }
        }
    }
}

/// Keep AUTO fuzziness exact at the public dataset-planning boundary.
///
/// Low-level index preparation already understands `fuzziness=None`, but a
/// partial dataset plan must prepare one vocabulary across indexed and current
/// unindexed rows. AUTO activation is deferred until OSS-2105 lands that
/// current-row preparation atomically. Until then, recursively rewrite AUTO to
/// exact while preserving explicit positive fuzziness.
pub fn apply_dataset_planner_auto_fuzziness_compatibility_gate(query: &mut FtsQuery) {
    match query {
        FtsQuery::Match(query) => {
            query.fuzziness.get_or_insert(0);
        }
        // combined_fields matches terms exactly; it has no fuzziness setting.
        FtsQuery::Phrase(_) | FtsQuery::CombinedFields(_) => {}
        FtsQuery::Boost(query) => {
            apply_dataset_planner_auto_fuzziness_compatibility_gate(&mut query.positive);
            apply_dataset_planner_auto_fuzziness_compatibility_gate(&mut query.negative);
        }
        FtsQuery::MultiMatch(query) => {
            for match_query in &mut query.match_queries {
                match_query.fuzziness.get_or_insert(0);
            }
        }
        FtsQuery::Boolean(query) => {
            for child in query
                .should
                .iter_mut()
                .chain(&mut query.must)
                .chain(&mut query.must_not)
            {
                apply_dataset_planner_auto_fuzziness_compatibility_gate(child);
            }
        }
    }
}

impl Scanner {
    pub(super) async fn fts_search_source(
        &self,
        filter_plan: &mut FilterPlan,
        query: &FullTextSearchQuery,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        log::trace!("source is an fts search");
        if self.include_deleted_rows {
            return Err(Error::invalid_input_source(
                "Cannot include deleted rows in an FTS search".into(),
            ));
        }

        // The source is an FTS search
        if self.prefilter {
            let source: Arc<dyn ExecutionPlan> = match &filter_plan.vector_filter() {
                Some(vector_query) => {
                    // Perform vector search first then rerank according to BM25 scores
                    let vector_plan = self
                        .vector_search(&filter_plan.expr_filter_plan, vector_query)
                        .await?;
                    self.fts_rerank(vector_plan, query).await?
                }
                None => self.fts(&filter_plan.expr_filter_plan, query).await?,
            };
            // If we are prefiltering then the fts node will take care of the filter
            filter_plan.disable_refine();
            Ok(source)
        } else {
            // If we are postfiltering then we can't use scalar indices for the filter
            // and will need to run the postfilter in memory
            filter_plan.make_refine_only();
            self.fts(&ExprFilterPlan::default(), query).await
        }
    }

    /// Convert element-document hits into the row-selection semantics required
    /// when FTS is used as a filter for another query.
    pub(super) fn deduplicate_fts_filter_rows(
        &self,
        input: Arc<dyn ExecutionPlan>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let schema = input.schema();
        let group_expr = vec![(
            expressions::col(ROW_ID, schema.as_ref())?,
            ROW_ID.to_string(),
        )];
        let input = Arc::new(RepartitionExec::try_new(
            input,
            Partitioning::RoundRobinBatch(1),
        )?);
        Ok(Arc::new(AggregateExec::try_new(
            AggregateMode::Single,
            PhysicalGroupBy::new_single(group_expr),
            Vec::new(),
            Vec::new(),
            input,
            schema,
        )?))
    }

    async fn fragments_covered_by_fts_leaf(
        &self,
        column: &str,
        document_granularity: DocumentGranularity,
        accum: &mut RoaringBitmap,
    ) -> Result<bool> {
        let Some(fragment_bitmap) =
            fts_index_fragment_bitmap(&self.dataset, column, document_granularity).await?
        else {
            return Ok(false);
        };
        *accum |= fragment_bitmap;

        Ok(true)
    }

    #[async_recursion]
    async fn fragments_covered_by_fts_query_helper(
        &self,
        query: &FtsQuery,
        accum: &mut RoaringBitmap,
    ) -> Result<bool> {
        match query {
            FtsQuery::Match(match_query) => {
                let document_granularity = match_query.document_granularity.ok_or_else(|| {
                    Error::internal("FTS Match query granularity was not resolved".to_string())
                })?;
                self.fragments_covered_by_fts_leaf(
                    match_query.column.as_ref().ok_or(Error::invalid_input(
                        "the column must be specified in the query".to_string(),
                    ))?,
                    document_granularity,
                    accum,
                )
                .await
            }
            FtsQuery::Boost(boost) => Ok(self
                .fragments_covered_by_fts_query_helper(&boost.negative, accum)
                .await?
                & self
                    .fragments_covered_by_fts_query_helper(&boost.positive, accum)
                    .await?),
            FtsQuery::MultiMatch(multi_match) => {
                for mq in &multi_match.match_queries {
                    let document_granularity = mq.document_granularity.ok_or_else(|| {
                        Error::internal(
                            "FTS MultiMatch query granularity was not resolved".to_string(),
                        )
                    })?;
                    if !self
                        .fragments_covered_by_fts_leaf(
                            mq.column.as_ref().ok_or(Error::invalid_input(
                                "the column must be specified in the query".to_string(),
                            ))?,
                            document_granularity,
                            accum,
                        )
                        .await?
                    {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            FtsQuery::Phrase(phrase_query) => {
                let document_granularity = phrase_query.document_granularity.ok_or_else(|| {
                    Error::internal("FTS Phrase query granularity was not resolved".to_string())
                })?;
                self.fragments_covered_by_fts_leaf(
                    phrase_query.column.as_ref().ok_or(Error::invalid_input(
                        "the column must be specified in the query".to_string(),
                    ))?,
                    document_granularity,
                    accum,
                )
                .await
            }
            FtsQuery::CombinedFields(combined) => {
                // A doc's fragment must be covered by every target column's index
                // for the prefilter to be exact, mirroring MultiMatch.
                for column in combined.column_names() {
                    if !self
                        .fragments_covered_by_fts_leaf(column, DocumentGranularity::Row, accum)
                        .await?
                    {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            FtsQuery::Boolean(bool_query) => {
                for query in bool_query
                    .must
                    .iter()
                    .chain(&bool_query.should)
                    .chain(&bool_query.must_not)
                {
                    if !self
                        .fragments_covered_by_fts_query_helper(query, accum)
                        .await?
                    {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
        }
    }

    async fn fragments_covered_by_fts_query(&self, query: &FtsQuery) -> Result<RoaringBitmap> {
        let all_fragments = self.get_fragments_as_bitmap();

        let mut referenced_fragments = RoaringBitmap::new();
        if !self
            .fragments_covered_by_fts_query_helper(query, &mut referenced_fragments)
            .await?
        {
            // One or more indices is missing the fragment bitmap, require all fragments in prefilter
            Ok(all_fragments)
        } else {
            // Fragments required for prefilter is intersection of index fragments and query fragments
            Ok(all_fragments & referenced_fragments)
        }
    }

    pub(super) fn fts_document_granularity(&self, query: &FtsQuery) -> Result<DocumentGranularity> {
        #[derive(Default)]
        struct TargetState {
            granularity: Option<DocumentGranularity>,
            list_element_field_id: Option<i32>,
        }

        fn add_leaf(
            schema: &lance_core::datatypes::Schema,
            column: Option<&str>,
            document_granularity: Option<DocumentGranularity>,
            state: &mut TargetState,
        ) -> Result<()> {
            let document_granularity = document_granularity.ok_or_else(|| {
                Error::internal("FTS query document granularity was not resolved".to_string())
            })?;
            if let Some(existing) = state.granularity
                && existing != document_granularity
            {
                return Err(Error::invalid_input(
                    "FTS queries cannot mix Row and ListElement document granularities".to_string(),
                ));
            }
            state.granularity = Some(document_granularity);

            let Some(column) = column else {
                if document_granularity.is_list_element() {
                    return Err(Error::invalid_input(
                        "ListElement FTS queries must explicitly specify a field path".to_string(),
                    ));
                }
                return Ok(());
            };
            let resolved = resolve_fts_field(schema, column, document_granularity)?;
            if document_granularity.is_list_element() {
                if let Some(existing) = state.list_element_field_id
                    && existing != resolved.final_field_id
                {
                    return Err(Error::invalid_input(
                        "all leaves in a ListElement FTS query must use the same final field"
                            .to_string(),
                    ));
                }
                state.list_element_field_id = Some(resolved.final_field_id);
            }
            Ok(())
        }

        fn visit(
            schema: &lance_core::datatypes::Schema,
            query: &FtsQuery,
            state: &mut TargetState,
        ) -> Result<()> {
            match query {
                FtsQuery::Match(query) => add_leaf(
                    schema,
                    query.column.as_deref(),
                    query.document_granularity,
                    state,
                ),
                FtsQuery::Phrase(query) => add_leaf(
                    schema,
                    query.column.as_deref(),
                    query.document_granularity,
                    state,
                ),
                FtsQuery::Boost(query) => {
                    visit(schema, &query.positive, state)?;
                    visit(schema, &query.negative, state)
                }
                FtsQuery::Boolean(query) => {
                    for child in query
                        .must
                        .iter()
                        .chain(&query.should)
                        .chain(&query.must_not)
                    {
                        visit(schema, child, state)?;
                    }
                    Ok(())
                }
                FtsQuery::MultiMatch(query) => {
                    for child in &query.match_queries {
                        if child
                            .document_granularity
                            .is_some_and(|value| value.is_list_element())
                        {
                            return Err(Error::not_supported(
                                "MultiMatch does not support ListElement document granularity"
                                    .to_string(),
                            ));
                        }
                        add_leaf(
                            schema,
                            child.column.as_deref(),
                            child.document_granularity,
                            state,
                        )?;
                    }
                    Ok(())
                }
                FtsQuery::CombinedFields(query) => {
                    // BM25F sums each target column's contribution for one row,
                    // so combined_fields is row-granular by construction. A
                    // target column that only has a list-element index was
                    // already rejected by
                    // `resolve_fts_query_document_granularity`.
                    for column in query.column_names() {
                        add_leaf(schema, Some(column), Some(DocumentGranularity::Row), state)?;
                    }
                    Ok(())
                }
            }
        }

        let mut state = TargetState::default();
        visit(self.dataset.schema(), query, &mut state)?;
        state.granularity.ok_or_else(|| {
            Error::invalid_input("FTS query must contain at least one leaf query".to_string())
        })
    }

    #[async_recursion]
    async fn resolve_fts_query_document_granularity(&self, query: FtsQuery) -> Result<FtsQuery> {
        match query {
            FtsQuery::Match(mut query) => {
                let column = query.column.as_deref().ok_or_else(|| {
                    Error::invalid_input("the column must be specified in the query".to_string())
                })?;
                query.document_granularity = Some(
                    resolve_query_document_granularity(
                        self.dataset.as_ref(),
                        column,
                        query.document_granularity,
                    )
                    .await?,
                );
                Ok(FtsQuery::Match(query))
            }
            FtsQuery::Phrase(mut query) => {
                let column = query.column.as_deref().ok_or_else(|| {
                    Error::invalid_input("the column must be specified in the query".to_string())
                })?;
                query.document_granularity = Some(
                    resolve_query_document_granularity(
                        self.dataset.as_ref(),
                        column,
                        query.document_granularity,
                    )
                    .await?,
                );
                Ok(FtsQuery::Phrase(query))
            }
            FtsQuery::Boost(mut query) => {
                query.positive = Box::new(
                    self.resolve_fts_query_document_granularity(*query.positive)
                        .await?,
                );
                query.negative = Box::new(
                    self.resolve_fts_query_document_granularity(*query.negative)
                        .await?,
                );
                Ok(FtsQuery::Boost(query))
            }
            FtsQuery::Boolean(mut query) => {
                let mut must = Vec::with_capacity(query.must.len());
                for child in query.must {
                    must.push(self.resolve_fts_query_document_granularity(child).await?);
                }
                let mut should = Vec::with_capacity(query.should.len());
                for child in query.should {
                    should.push(self.resolve_fts_query_document_granularity(child).await?);
                }
                let mut must_not = Vec::with_capacity(query.must_not.len());
                for child in query.must_not {
                    must_not.push(self.resolve_fts_query_document_granularity(child).await?);
                }
                query.must = must;
                query.should = should;
                query.must_not = must_not;
                Ok(FtsQuery::Boolean(query))
            }
            FtsQuery::MultiMatch(mut query) => {
                for child in &mut query.match_queries {
                    let column = child.column.as_deref().ok_or_else(|| {
                        Error::invalid_input(
                            "the column must be specified in the query".to_string(),
                        )
                    })?;
                    child.document_granularity = Some(
                        resolve_query_document_granularity(
                            self.dataset.as_ref(),
                            column,
                            child.document_granularity,
                        )
                        .await?,
                    );
                }
                Ok(FtsQuery::MultiMatch(query))
            }
            FtsQuery::CombinedFields(query) => {
                // There is nothing to resolve: a combined_fields query carries no
                // granularity because BM25F only makes sense over row documents.
                // Reject a target column that cannot supply them.
                for column in query.column_names() {
                    validate_combined_fields_target_column(self.dataset.as_ref(), column).await?;
                }
                Ok(FtsQuery::CombinedFields(query))
            }
        }
    }

    fn set_missing_query_granularity(
        query: &mut FtsQuery,
        document_granularity: DocumentGranularity,
    ) {
        match query {
            FtsQuery::Match(query) => {
                query
                    .document_granularity
                    .get_or_insert(document_granularity);
            }
            FtsQuery::Phrase(query) => {
                query
                    .document_granularity
                    .get_or_insert(document_granularity);
            }
            FtsQuery::Boost(query) => {
                Self::set_missing_query_granularity(&mut query.positive, document_granularity);
                Self::set_missing_query_granularity(&mut query.negative, document_granularity);
            }
            FtsQuery::Boolean(query) => {
                for child in query
                    .must
                    .iter_mut()
                    .chain(&mut query.should)
                    .chain(&mut query.must_not)
                {
                    Self::set_missing_query_granularity(child, document_granularity);
                }
            }
            FtsQuery::MultiMatch(query) => {
                for child in &mut query.match_queries {
                    child
                        .document_granularity
                        .get_or_insert(document_granularity);
                }
            }
            // Row-granular by construction, and it carries no field to fill in.
            FtsQuery::CombinedFields(_) => {}
        }
    }

    fn query_requests_list_element(query: &FtsQuery) -> bool {
        match query {
            FtsQuery::Match(query) => query
                .document_granularity
                .is_some_and(DocumentGranularity::is_list_element),
            FtsQuery::Phrase(query) => query
                .document_granularity
                .is_some_and(DocumentGranularity::is_list_element),
            FtsQuery::Boost(query) => {
                Self::query_requests_list_element(&query.positive)
                    || Self::query_requests_list_element(&query.negative)
            }
            FtsQuery::Boolean(query) => query
                .must
                .iter()
                .chain(&query.should)
                .chain(&query.must_not)
                .any(Self::query_requests_list_element),
            FtsQuery::MultiMatch(query) => query.match_queries.iter().any(|query| {
                query
                    .document_granularity
                    .is_some_and(DocumentGranularity::is_list_element)
            }),
            FtsQuery::CombinedFields(_) => false,
        }
    }

    pub(super) async fn resolve_full_text_search_query(
        &self,
        query: &FullTextSearchQuery,
    ) -> Result<FullTextSearchQuery> {
        let mut resolved = query.clone();
        normalize_fts_zero_boosts(&mut resolved.query);
        if resolved.query.is_missing_column() {
            if Self::query_requests_list_element(&resolved.query) {
                return Err(Error::invalid_input(
                    "ListElement FTS queries must explicitly specify a field path".to_string(),
                ));
            }
            let indexed_columns = fts_indexed_columns(self.dataset.clone()).await?;
            resolved.query = fill_fts_query_column(&resolved.query, &indexed_columns, false)?;
            Self::set_missing_query_granularity(&mut resolved.query, DocumentGranularity::Row);
        }
        apply_dataset_planner_auto_fuzziness_compatibility_gate(&mut resolved.query);
        resolved.query = self
            .resolve_fts_query_document_granularity(resolved.query)
            .await?;
        self.fts_document_granularity(&resolved.query)?;
        Ok(resolved)
    }

    // Create an execution plan to do full text search
    pub(super) async fn fts(
        &self,
        filter_plan: &ExprFilterPlan,
        query: &FullTextSearchQuery,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let mut params = query.params();
        if params.limit.is_none() {
            let search_limit = match (self.limit, self.offset) {
                (Some(limit), Some(offset)) => Some((limit + offset) as usize),
                (Some(limit), None) => Some(limit as usize),
                (None, Some(_)) => None, // No limit but has offset - fetch all and let limit_node handle
                (None, None) => None,
            };
            params = params.with_limit(search_limit);
        }
        let query = &query.query;
        validate_fts_query_contract(query)?;

        // TODO: Could maybe walk the query here to find all the indices that will be
        // involved in the query to calculate a more accuarate required_fragments than
        // get_fragments_as_bitmap but this is safe for now.
        let prefilter_source = self
            .prefilter_source(
                filter_plan,
                self.fragments_covered_by_fts_query(query).await?,
            )
            .await?;
        // Data overlay masking blocks stale rows from indexed leaves and re-evaluates only those
        // rows from their current values on the flat-text path.
        let fts_exec = self
            .plan_fts(query, &params, filter_plan, &prefilter_source)
            .await?;
        Ok(fts_exec)
    }

    async fn plan_compound_scorer(
        &self,
        query: &FtsQuery,
        params: &FtsSearchParams,
        filter_plan: &ExprFilterPlan,
        prefilter_source: &PreFilterSource,
        document_granularity: DocumentGranularity,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let columns = collect_fts_columns_in_order(query);
        if columns.is_empty() {
            return Ok(None);
        }
        let cross_column = columns.len() > 1;
        if cross_column && params.limit.is_none() {
            // Candidate-driven cross-column execution requires a bounded top-k
            // collector. The existing DataFusion plan remains the exact path
            // for callers that request the full result set.
            return Ok(None);
        }

        let target_fragments: &[Fragment] = self
            .fragments
            .as_deref()
            .unwrap_or_else(|| self.dataset.fragments());
        if target_fragments.is_empty() {
            return Ok(None);
        }
        let mut phrase_columns = HashSet::new();
        collect_phrase_columns(query, &mut phrase_columns);
        // Query-local residual scoring intentionally reuses committed-index
        // BM25 statistics. Matching remains exact for the supported leaf
        // shapes, but ranking is approximate until the appended rows are
        // incorporated into a persistent index.
        let allow_indexed_stats_residual = !cross_column
            && !self.fast_search
            && self.fragments.is_none()
            && filter_plan.is_empty()
            && self.external_row_mask.is_none()
            && params.limit.is_some()
            && document_granularity == DocumentGranularity::Row
            && target_fragments
                .iter()
                .all(|fragment| fragment.deletion_file.is_none())
            && supports_indexed_stats_residual_compound(query);

        let segment_groups = futures::future::try_join_all(columns.into_iter().map(|column| {
            let phrase_columns = &phrase_columns;
            async move {
                let index = self
                    .dataset
                    .load_scalar_index(
                        IndexCriteria::default()
                            .for_column(&column)
                            .supports_fts()
                            .with_fts_document_granularity(document_granularity),
                    )
                    .await?;
                let Some(index) = index else {
                    return Ok(None);
                };

                let (unindexed_fragments, overlay_plan) = futures::future::try_join(
                    self.dataset.unindexed_fragments(&index.name),
                    self.fts_overlay_plan(&column, document_granularity, target_fragments),
                )
                .await?;
                let unindexed_fragments = self.retain_target_fragments(unindexed_fragments);
                let has_bounded_residual = allow_indexed_stats_residual
                    && has_bounded_query_local_residual_rows(&unindexed_fragments);
                if !unindexed_fragments.is_empty()
                    && (!self.fast_search || unindexed_fragments.len() == target_fragments.len())
                    && !(has_bounded_residual
                        && unindexed_fragments.len() < target_fragments.len())
                {
                    // Flat and posting-backed leaves do not share a document
                    // domain, so preserve the exact fallback for partial index
                    // coverage. Fast search deliberately excludes unindexed
                    // fragments, so its indexed-only domain remains valid for
                    // the compound scorer when at least one target fragment is
                    // indexed.
                    return Ok(None);
                }
                let segments = match overlay_plan {
                    FtsOverlayPlan::Unchanged(Some(segments)) => segments,
                    FtsOverlayPlan::Unchanged(None) => {
                        load_segments(&self.dataset, &column, document_granularity)
                            .await?
                            .ok_or_else(|| {
                                Error::invalid_input(format!(
                                    "No Inverted index found for column {column}"
                                ))
                            })?
                    }
                    FtsOverlayPlan::RowLevel { .. } | FtsOverlayPlan::FullScan => return Ok(None),
                };
                if has_bounded_residual && !unindexed_fragments.is_empty() {
                    if !has_complete_hybrid_fts_coverage(
                        &segments,
                        &unindexed_fragments,
                        target_fragments,
                    ) {
                        return Ok(None);
                    }
                    if segments.is_empty() {
                        return Err(Error::internal(
                            "hybrid compound FTS requires one indexed segment",
                        ));
                    }
                    // Preserve the established semantic mismatch error before
                    // constructing query-local postings with the same tokenizer.
                    load_segment_details(&self.dataset, &column, &segments).await?;
                }

                if cross_column {
                    let details = futures::future::try_join_all(
                        segments.iter().map(|segment| {
                            load_physical_fts_details(&self.dataset, &column, segment)
                        }),
                    )
                    .await?;
                    if phrase_columns.contains(&column)
                        && details.iter().any(|details| !details.with_position)
                    {
                        return Err(Error::invalid_input(
                            "position is not found but required for phrase queries, try recreating the index with position"
                                .to_string(),
                        ));
                    }
                    let all_modern = details.iter().all(|details| {
                        matches!(
                            details.posting_format_version,
                            Some(INVERTED_INDEX_VERSION_V2 | INVERTED_INDEX_VERSION_V3)
                        )
                    });
                    if !all_modern {
                        return Ok(None);
                    }
                } else if phrase_columns.contains(&column) {
                    let details = load_segment_details(&self.dataset, &column, &segments).await?;
                    if !details.with_position {
                        return Err(Error::invalid_input(
                            "position is not found but required for phrase queries, try recreating the index with position"
                                .to_string(),
                        ));
                    }
                }

                Ok(Some((column, segments, unindexed_fragments)))
            }
        }))
        .await?;
        let Some(segment_groups) = segment_groups.into_iter().collect::<Option<Vec<_>>>() else {
            return Ok(None);
        };

        if !cross_column {
            let (column, segments, unindexed_fragments) =
                segment_groups.into_iter().next().ok_or_else(|| {
                    Error::internal("compound scorer requires one column".to_string())
                })?;
            if allow_indexed_stats_residual && !unindexed_fragments.is_empty() {
                let resolved =
                    resolve_fts_field(self.dataset.schema(), &column, document_granularity)?;
                let scan_column = if resolved.has_lists() {
                    resolved.root_column.clone()
                } else {
                    resolved.canonical_path.clone()
                };
                let scan_projection = self
                    .dataset
                    .empty_projection()
                    .with_row_id()
                    .union_columns(&[scan_column], OnMissing::Error)?;
                let PlannedFilteredScan { plan, .. } = self
                    .filtered_read(
                        &ExprFilterPlan::default(),
                        scan_projection,
                        /* make_deletions_null */ false,
                        Some(Arc::new(unindexed_fragments)),
                        None,
                        /* is_prefilter */ true,
                        None,
                    )
                    .await?;
                return Ok(Some(Arc::new(HybridCompoundQueryExec::new(
                    self.dataset.clone(),
                    query.clone(),
                    params.clone(),
                    column,
                    segments,
                    plan,
                ))));
            }
            return Ok(Some(Arc::new(
                CompoundQueryExec::new_with_segments(
                    self.dataset.clone(),
                    query.clone(),
                    params.clone(),
                    prefilter_source.clone(),
                    segments,
                )
                .with_external_mask(self.external_row_mask.clone()),
            )));
        }

        let mut coverage_groups = segment_groups.iter();
        let Some((_, _, first_unindexed_fragments)) = coverage_groups.next() else {
            return Ok(None);
        };
        let first_unindexed_fragment_ids = first_unindexed_fragments
            .iter()
            .map(|fragment| fragment.id as u32)
            .collect::<RoaringBitmap>();
        if coverage_groups.any(|(_, _, unindexed_fragments)| {
            unindexed_fragments
                .iter()
                .map(|fragment| fragment.id as u32)
                .collect::<RoaringBitmap>()
                != first_unindexed_fragment_ids
        }) {
            // The cross-column scorer builds one shared prefilter. If column
            // coverage differs, that prefilter's union can re-admit stale
            // postings from a fragment invalidated only for another column.
            // Keep the field-local fallback, which preserves each column's
            // own index domain.
            return Ok(None);
        }
        let segment_groups = segment_groups
            .into_iter()
            .map(|(column, segments, _)| (column, segments))
            .collect();
        let exec = CrossColumnCompoundQueryExec::new_with_segments(
            self.dataset.clone(),
            query.clone(),
            params.clone(),
            prefilter_source.clone(),
            segment_groups,
        )?
        .with_external_mask(self.external_row_mask.clone());
        Ok(Some(Arc::new(exec)))
    }

    async fn plan_fts(
        &self,
        query: &FtsQuery,
        params: &FtsSearchParams,
        filter_plan: &ExprFilterPlan,
        prefilter_source: &PreFilterSource,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let document_granularity = self.fts_document_granularity(query)?;
        if !document_granularity.is_list_element()
            && supports_compound_scorer(query)
            && let Some(plan) = self
                .plan_compound_scorer(
                    query,
                    params,
                    filter_plan,
                    prefilter_source,
                    document_granularity,
                )
                .await?
        {
            return Ok(plan);
        }

        // Unsupported, unbounded, partial-index, and overlay-backed cross-column
        // shapes retain the exact DataFusion fallback.
        let plan: Arc<dyn ExecutionPlan> = match query {
            FtsQuery::Match(query) => {
                self.plan_match_query(query, params, filter_plan, prefilter_source)
                    .await?
            }
            FtsQuery::Phrase(query) => {
                self.plan_phrase_query(query, params, filter_plan, prefilter_source)
                    .await?
            }

            FtsQuery::Boost(query) => {
                // for boost query, we need to erase the limit so that we can find
                // the documents that are not in the top-k results of the positive query,
                // but in the final top-k results.
                let unlimited_params = params.clone().with_limit(None);
                let positive_exec = Box::pin(self.plan_fts(
                    &query.positive,
                    &unlimited_params,
                    filter_plan,
                    prefilter_source,
                ));
                let negative_exec = Box::pin(self.plan_fts(
                    &query.negative,
                    &unlimited_params,
                    filter_plan,
                    prefilter_source,
                ));
                let (positive_exec, negative_exec) =
                    futures::future::try_join(positive_exec, negative_exec).await?;
                Arc::new(BoostQueryExec::new(
                    query.clone(),
                    params.clone(),
                    positive_exec,
                    negative_exec,
                ))
            }

            FtsQuery::MultiMatch(query) => {
                // A top-level cross-column MultiMatch scores each field independently and takes
                // the maximum score for each row. A field's bounded compound top-k is therefore
                // sufficient to determine the global top-k independently of the other fields.
                // Preserve that bounded plan for every eligible field, while planning only
                // partial-index and overlay-backed fields through the exhaustive leaf fallback.
                let unlimited_params = params.clone().with_limit(None);
                let can_use_bounded_compound =
                    !document_granularity.is_list_element() && params.limit.is_some();
                let field_prefilter_sources =
                    prefilter_source.shared_for_multimatch_fields(query.match_queries.len());
                let children = futures::future::try_join_all(
                    query
                        .match_queries
                        .iter()
                        .zip(field_prefilter_sources.iter())
                        .map(|(match_query, field_prefilter_source)| {
                            let unlimited_params = &unlimited_params;
                            async move {
                                if can_use_bounded_compound {
                                    let child_query = FtsQuery::Match(match_query.clone());
                                    if let Some(plan) = self
                                        .plan_compound_scorer(
                                            &child_query,
                                            params,
                                            filter_plan,
                                            field_prefilter_source,
                                            document_granularity,
                                        )
                                        .await?
                                    {
                                        return Ok(plan);
                                    }
                                }

                                self.plan_match_query(
                                    match_query,
                                    unlimited_params,
                                    filter_plan,
                                    field_prefilter_source,
                                )
                                .await
                            }
                        }),
                )
                .await?;

                let schema = children[0].schema();
                let group_expr = vec![(
                    expressions::col(ROW_ID, schema.as_ref())?,
                    ROW_ID.to_string(),
                )];

                let fts_node = UnionExec::try_new(children)?;
                let fts_node = Arc::new(RepartitionExec::try_new(
                    fts_node,
                    Partitioning::RoundRobinBatch(1),
                )?);
                // dedup by row_id and return the max score as final score
                let fts_node = Arc::new(AggregateExec::try_new(
                    AggregateMode::Single,
                    PhysicalGroupBy::new_single(group_expr),
                    vec![Arc::new(
                        datafusion_physical_expr::aggregate::AggregateExprBuilder::new(
                            functions_aggregate::min_max::max_udaf(),
                            vec![expressions::col(SCORE_COL, &schema)?],
                        )
                        .schema(schema.clone())
                        .alias(SCORE_COL)
                        .build()?,
                    )],
                    vec![None],
                    fts_node,
                    schema,
                )?);
                let sort_exprs = Self::fts_score_sort_exprs(fts_node.schema().as_ref())?;

                // `params.limit` is the recursive planning contract. Compound
                // parents pass `None` when they require every candidate.
                Arc::new(SortExec::new(sort_exprs.into(), fts_node).with_fetch(params.limit))
            }
            FtsQuery::CombinedFields(query) => {
                self.plan_combined_fields_query(query, params, filter_plan, prefilter_source)
                    .await?
            }
            FtsQuery::Boolean(query) => {
                // TODO: rewrite the query for better performance

                // we need to remove the limit from the params,
                // so that we won't miss possible matches
                let unlimited_params = params.clone().with_limit(None);

                let mut should = Vec::with_capacity(query.should.len());
                for subquery in &query.should {
                    should.push(
                        Box::pin(self.plan_fts(
                            subquery,
                            &unlimited_params,
                            filter_plan,
                            prefilter_source,
                        ))
                        .await?,
                    );
                }
                let mut must = Vec::with_capacity(query.must.len());
                for subquery in &query.must {
                    must.push(
                        Box::pin(self.plan_fts(
                            subquery,
                            &unlimited_params,
                            filter_plan,
                            prefilter_source,
                        ))
                        .await?,
                    );
                }
                let mut must_not = Vec::with_capacity(query.must_not.len());
                for subquery in &query.must_not {
                    must_not.push(
                        Box::pin(self.plan_fts(
                            subquery,
                            &unlimited_params,
                            filter_plan,
                            prefilter_source,
                        ))
                        .await?,
                    );
                }

                let boolean_schema = fts_schema(document_granularity);
                let should = build_boolean_query_children_with_schema(
                    BoolSlot::Should,
                    should,
                    boolean_schema.clone(),
                )?
                .ok_or_else(|| {
                    Error::internal(
                        "boolean should planning returned no execution plan".to_string(),
                    )
                })?;
                let must = build_boolean_query_children_with_schema(
                    BoolSlot::Must,
                    must,
                    boolean_schema.clone(),
                )?;
                let must_not = build_boolean_query_children_with_schema(
                    BoolSlot::MustNot,
                    must_not,
                    boolean_schema,
                )?
                .ok_or_else(|| {
                    Error::internal(
                        "boolean must-not planning returned no execution plan".to_string(),
                    )
                })?;

                if query.should.is_empty() && must.is_none() {
                    return Err(Error::invalid_input(
                        "boolean query must have at least one should/must query".to_string(),
                    ));
                }

                Arc::new(BooleanQueryExec::new(
                    query.clone(),
                    params.clone(),
                    should,
                    must,
                    must_not,
                ))
            }
        };

        Ok(plan)
    }

    async fn plan_phrase_query(
        &self,
        query: &PhraseQuery,
        params: &FtsSearchParams,
        filter_plan: &ExprFilterPlan,
        prefilter_source: &PreFilterSource,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let column = query.column.clone().ok_or(Error::invalid_input(
            "the column must be specified in the query".to_string(),
        ))?;
        let document_granularity = query.document_granularity.ok_or_else(|| {
            Error::internal("FTS Phrase query granularity was not resolved".to_string())
        })?;
        resolve_fts_field(self.dataset.schema(), &column, document_granularity)?;
        let output_schema = fts_schema(document_granularity);
        let index = self
            .dataset
            .load_scalar_index(
                IndexCriteria::default()
                    .for_column(&column)
                    .supports_fts()
                    .with_fts_document_granularity(document_granularity),
            )
            .await?;
        let target_fragments: &[Fragment] = self
            .fragments
            .as_deref()
            .unwrap_or_else(|| self.dataset.fragments());
        if self.fragments.as_ref().is_some_and(Vec::is_empty) {
            return Ok(Arc::new(EmptyExec::new(output_schema)));
        }
        let flat_query = MatchQuery::new(query.terms.clone())
            .with_column(Some(column.clone()))
            .with_operator(Operator::And)
            .with_document_granularity(document_granularity);
        let flat_params = params.clone().with_phrase_slop(Some(query.slop));

        let (phrase_plan, flat_phrase_plan) = match &index {
            Some(index) => {
                let unindexed_fragments = self
                    .retain_target_fragments(self.dataset.unindexed_fragments(&index.name).await?);
                if !target_fragments.is_empty()
                    && unindexed_fragments.len() == target_fragments.len()
                {
                    if self.fast_search {
                        return Ok(Arc::new(EmptyExec::new(output_schema)));
                    }
                    let flat_phrase_plan = self
                        .plan_flat_match_query(
                            unindexed_fragments,
                            HashMap::new(),
                            &flat_query,
                            &flat_params,
                            filter_plan,
                            None,
                        )
                        .await?;
                    return Self::combine_fts_leaf_plans(None, Some(flat_phrase_plan), params);
                }

                let (stale_rows, preset_segments) = match self
                    .fts_overlay_plan(&column, document_granularity, target_fragments)
                    .await?
                {
                    FtsOverlayPlan::Unchanged(segments) => (HashMap::new(), segments),
                    FtsOverlayPlan::RowLevel {
                        stale_rows,
                        segments,
                    } => (stale_rows, Some(segments)),
                    FtsOverlayPlan::FullScan => {
                        if self.fast_search {
                            return Ok(Arc::new(EmptyExec::new(output_schema)));
                        }
                        let flat_phrase_plan = self
                            .plan_flat_match_query(
                                target_fragments.to_vec(),
                                HashMap::new(),
                                &flat_query,
                                &flat_params,
                                filter_plan,
                                None,
                            )
                            .await?;
                        return Self::combine_fts_leaf_plans(None, Some(flat_phrase_plan), params);
                    }
                };
                let overlay_block = self.stale_rows_block_mask(&stale_rows).await?;
                let segments = match preset_segments {
                    Some(segments) => segments,
                    None => load_segments(&self.dataset, &column, document_granularity)
                        .await?
                        .ok_or_else(|| {
                            Error::internal(format!(
                                "FTS metadata routed column {column} without loadable segments"
                            ))
                        })?,
                };
                let details = load_segment_details(&self.dataset, &column, &segments).await?;
                if !details.with_position {
                    return Err(Error::invalid_input("position is not found but required for phrase queries, try recreating the index with position"
                        .to_string()));
                }

                let has_flat_path = !self.fast_search
                    && (!unindexed_fragments.is_empty() || !stale_rows.is_empty());
                let shared_scorer = (has_flat_path && document_granularity.is_list_element())
                    .then(|| Arc::new(SharedFtsScorer::new()));
                let mut phrase_exec = PhraseQueryExec::new_with_segments_and_document_granularity(
                    self.dataset.clone(),
                    query.clone(),
                    params.clone(),
                    prefilter_source.clone(),
                    segments,
                    document_granularity,
                );
                if let Some(overlay_block) = overlay_block {
                    phrase_exec = phrase_exec.with_overlay_block(overlay_block);
                }
                if let Some(shared_scorer) = &shared_scorer {
                    phrase_exec = phrase_exec.with_shared_scorer(shared_scorer.clone());
                }
                phrase_exec = phrase_exec.with_external_mask(self.external_row_mask.clone());
                let phrase_plan = Some(Arc::new(phrase_exec) as Arc<dyn ExecutionPlan>);
                let flat_phrase_plan = if has_flat_path {
                    Some(
                        self.plan_flat_match_query(
                            unindexed_fragments,
                            stale_rows,
                            &flat_query,
                            &flat_params,
                            filter_plan,
                            shared_scorer,
                        )
                        .await?,
                    )
                } else {
                    None
                };
                (phrase_plan, flat_phrase_plan)
            }
            None => {
                if target_fragments.is_empty() {
                    return Ok(Arc::new(EmptyExec::new(output_schema)));
                }
                if self.fast_search {
                    return Ok(Arc::new(EmptyExec::new(output_schema)));
                }
                let flat_phrase_plan = self
                    .plan_flat_match_query(
                        target_fragments.to_vec(),
                        HashMap::new(),
                        &flat_query,
                        &flat_params,
                        filter_plan,
                        None,
                    )
                    .await?;
                (None, Some(flat_phrase_plan))
            }
        };

        Self::combine_fts_leaf_plans(phrase_plan, flat_phrase_plan, params)
    }

    async fn plan_match_query(
        &self,
        query: &MatchQuery,
        params: &FtsSearchParams,
        filter_plan: &ExprFilterPlan,
        prefilter_source: &PreFilterSource,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let column = query
            .column
            .as_ref()
            .ok_or(Error::invalid_input(
                "the column must be specified in the query".to_string(),
            ))?
            .clone();
        let document_granularity = query.document_granularity.ok_or_else(|| {
            Error::internal("FTS Match query granularity was not resolved".to_string())
        })?;
        resolve_fts_field(self.dataset.schema(), &column, document_granularity)?;
        let output_schema = fts_schema(document_granularity);

        let index = self
            .dataset
            .load_scalar_index(
                IndexCriteria::default()
                    .for_column(&column)
                    .supports_fts()
                    .with_fts_document_granularity(document_granularity),
            )
            .await?;

        // Get target fragments
        let target_fragments: &[Fragment] = self
            .fragments
            .as_deref()
            .unwrap_or_else(|| self.dataset.fragments());
        if self.fragments.as_ref().is_some_and(Vec::is_empty) {
            return Ok(Arc::new(EmptyExec::new(output_schema)));
        }

        let (match_plan, flat_match_plan) = match &index {
            Some(index) => {
                let unindexed_fragments = self
                    .retain_target_fragments(self.dataset.unindexed_fragments(&index.name).await?);
                if !target_fragments.is_empty()
                    && unindexed_fragments.len() == target_fragments.len()
                {
                    if self.fast_search {
                        return Ok(Arc::new(EmptyExec::new(output_schema)));
                    }
                    let flat_match_plan = self
                        .plan_flat_match_query(
                            unindexed_fragments,
                            HashMap::new(),
                            query,
                            params,
                            filter_plan,
                            None,
                        )
                        .await?;
                    return Self::combine_fts_leaf_plans(None, Some(flat_match_plan), params);
                }

                let (stale_rows, preset_segments) = match self
                    .fts_overlay_plan(&column, document_granularity, target_fragments)
                    .await?
                {
                    FtsOverlayPlan::Unchanged(segments) => (HashMap::new(), segments),
                    FtsOverlayPlan::RowLevel {
                        stale_rows,
                        segments,
                    } => (stale_rows, Some(segments)),
                    FtsOverlayPlan::FullScan => {
                        if self.fast_search {
                            return Ok(Arc::new(EmptyExec::new(output_schema)));
                        }
                        let flat_match_plan = self
                            .plan_flat_match_query(
                                target_fragments.to_vec(),
                                HashMap::new(),
                                query,
                                params,
                                filter_plan,
                                None,
                            )
                            .await?;
                        return Self::combine_fts_leaf_plans(None, Some(flat_match_plan), params);
                    }
                };
                let overlay_block = self.stale_rows_block_mask(&stale_rows).await?;
                let has_flat_path = !self.fast_search
                    && (!unindexed_fragments.is_empty() || !stale_rows.is_empty());
                let shared_scorer = (has_flat_path && document_granularity.is_list_element())
                    .then(|| Arc::new(SharedFtsScorer::new()));
                let mut match_exec = match preset_segments {
                    Some(segments) => MatchQueryExec::new_with_segments_and_document_granularity(
                        self.dataset.clone(),
                        query.clone(),
                        params.clone(),
                        prefilter_source.clone(),
                        segments,
                        document_granularity,
                    ),
                    None => MatchQueryExec::new_with_document_granularity(
                        self.dataset.clone(),
                        query.clone(),
                        params.clone(),
                        prefilter_source.clone(),
                        document_granularity,
                    ),
                };
                if let Some(overlay_block) = overlay_block {
                    match_exec = match_exec.with_overlay_block(overlay_block);
                }
                if let Some(shared_scorer) = &shared_scorer {
                    match_exec = match_exec.with_shared_scorer(shared_scorer.clone());
                }
                match_exec = match_exec.with_external_mask(self.external_row_mask.clone());
                let match_plan = Some(Arc::new(match_exec) as Arc<dyn ExecutionPlan>);
                let flat_match_plan = if has_flat_path {
                    Some(
                        self.plan_flat_match_query(
                            unindexed_fragments,
                            stale_rows,
                            query,
                            params,
                            filter_plan,
                            shared_scorer,
                        )
                        .await?,
                    )
                } else {
                    None
                };
                (match_plan, flat_match_plan)
            }
            None => {
                if target_fragments.is_empty() {
                    return Ok(Arc::new(EmptyExec::new(output_schema)));
                }
                if self.fast_search {
                    return Ok(Arc::new(EmptyExec::new(output_schema)));
                }
                // No index: flat search all target fragments
                let flat_match_plan = self
                    .plan_flat_match_query(
                        target_fragments.to_vec(),
                        HashMap::new(),
                        query,
                        params,
                        filter_plan,
                        None,
                    )
                    .await?;
                (None, Some(flat_match_plan))
            }
        };

        Self::combine_fts_leaf_plans(match_plan, flat_match_plan, params)
    }

    fn combine_fts_leaf_plans(
        indexed_plan: Option<Arc<dyn ExecutionPlan>>,
        flat_plan: Option<Arc<dyn ExecutionPlan>>,
        params: &FtsSearchParams,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let plan = match (indexed_plan, flat_plan) {
            (Some(indexed_plan), Some(flat_plan)) => {
                UnionExec::try_new(vec![indexed_plan, flat_plan])?
            }
            (Some(indexed_plan), None) => return Ok(indexed_plan),
            (None, Some(flat_plan)) if params.limit.is_none() => return Ok(flat_plan),
            (None, Some(flat_plan)) => flat_plan,
            (None, None) => {
                return Err(Error::internal(
                    "FTS leaf planning produced neither an indexed nor a flat plan".to_string(),
                ));
            }
        };
        let plan = Arc::new(RepartitionExec::try_new(
            plan,
            Partitioning::RoundRobinBatch(1),
        )?);
        let sort_expr = PhysicalSortExpr {
            expr: expressions::col(SCORE_COL, plan.schema().as_ref())?,
            options: SortOptions {
                descending: true,
                nulls_first: false,
            },
        };
        Ok(Arc::new(
            SortExec::new([sort_expr].into(), plan).with_fetch(params.limit),
        ))
    }

    /// Plan match query on unindexed fragments
    async fn plan_flat_match_query(
        &self,
        fragments: Vec<Fragment>,
        stale_rows: HashMap<u32, RoaringBitmap>,
        query: &MatchQuery,
        params: &FtsSearchParams,
        filter_plan: &ExprFilterPlan,
        shared_scorer: Option<Arc<SharedFtsScorer>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let column = query
            .column
            .as_ref()
            .ok_or(Error::invalid_input(
                "the column must be specified in the query".to_string(),
            ))?
            .clone();
        let document_granularity = query.document_granularity.ok_or_else(|| {
            Error::internal("FTS Match query granularity was not resolved".to_string())
        })?;
        let resolved = resolve_fts_field(self.dataset.schema(), &column, document_granularity)?;
        let scan_column = if resolved.has_lists() {
            resolved.root_column.clone()
        } else {
            resolved.canonical_path.clone()
        };
        let document_column = if resolved.has_lists() {
            VALUE_COLUMN_NAME.to_string()
        } else {
            resolved.canonical_path.clone()
        };
        let mut plan = self
            .plan_uncovered_rows_scan(fragments, stale_rows, vec![scan_column], filter_plan)
            .await?;
        if resolved.has_lists() {
            plan = Arc::new(FtsDocumentExec::new(plan, resolved.clone()));
        } else {
            plan = self.ensure_column_alias(plan, &document_column)?;
        }
        let mut flat_match_plan = FlatMatchQueryExec::new_with_document_granularity(
            self.dataset.clone(),
            query.clone(),
            params.clone(),
            plan,
            document_granularity,
            document_column,
        );
        if let Some(shared_scorer) = shared_scorer {
            flat_match_plan = flat_match_plan.with_shared_scorer(shared_scorer);
        }
        let flat_match_plan: Arc<dyn ExecutionPlan> = Arc::new(flat_match_plan);
        // Unindexed fragments and stale rows never reach the index-side prefilter,
        // so apply the external row-address mask to the flat FTS results here
        // (mirrors the ANN flat branch). Applied before the caller's top-k so
        // masked-out rows do not consume result slots.
        if let Some(mask) = self.external_row_mask.clone() {
            return Ok(Arc::new(RowAddrMaskFilterExec::new(flat_match_plan, mask)));
        }
        Ok(flat_match_plan)
    }

    /// Sort keys for an FTS plan whose scores come from more than one source, or
    /// from a source that does not sort.
    ///
    /// `row_id ASC` is a required second key, not a nicety: a score-only sort over
    /// concurrently read sources breaks ties by arrival order, so equal-scoring
    /// documents would come back in a different order run to run and
    /// `limit`/`offset` pagination could skip and repeat rows. `combined_fields`
    /// needs it for the same reason: `combined_fields_search` visits candidates in
    /// ascending row-id order, so its top-k is reproducible, but it does not order
    /// ties itself (`ScoredDoc` compares on score alone), so this sort is what
    /// decides which of two equal-scoring rows comes first.
    fn fts_score_sort_exprs(schema: &ArrowSchema) -> Result<[PhysicalSortExpr; 2]> {
        Ok([
            PhysicalSortExpr {
                expr: expressions::col(SCORE_COL, schema)?,
                options: SortOptions {
                    descending: true,
                    nulls_first: false,
                },
            },
            PhysicalSortExpr {
                expr: expressions::col(ROW_ID, schema)?,
                options: SortOptions {
                    descending: false,
                    nulls_first: false,
                },
            },
        ])
    }

    /// Plan a cross-field (BM25F) `combined_fields` query.
    ///
    /// Unlike a single-column match, coverage here is per column: a BM25F score is
    /// only complete when every target column's index covers the row's fragment,
    /// because `dl'` sums each column's document length and the per-column lookup
    /// (`AddressKeyedDocuments::doc_length_at` over that partition's `DocLengths`)
    /// returns 0 for a row that column's index holds no document for. So the
    /// fragments the index scan must not touch are the union of the per-column
    /// unindexed sets, not their intersection: a fragment indexed for `title` but
    /// not `body` would otherwise be emitted with a partial `tf'`/`dl'`.
    ///
    /// Those fragments are routed to [`FlatCombinedFieldsExec`] instead, and the
    /// two sides are unioned and re-sorted by score, mirroring
    /// [`Self::plan_match_query`].
    ///
    /// Rows whose index entries a newer data overlay made stale, in any target
    /// column, are routed the same way but one row at a time: the indexed scan
    /// blocks them, so it neither returns a pre-overlay hit nor hides a new one,
    /// and the flat side takes them by address. See [`Self::fts_overlay_plan`].
    async fn plan_combined_fields_query(
        &self,
        query: &CombinedFieldsQuery,
        params: &FtsSearchParams,
        filter_plan: &ExprFilterPlan,
        prefilter_source: &PreFilterSource,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let target_fragments: &[Fragment] = self
            .fragments
            .as_deref()
            .unwrap_or_else(|| self.dataset.fragments());
        // An explicitly empty fragment list selects no rows. The prefilter would
        // already reject every row, but without this the fully covered branch below
        // still opens the segments, builds the scorer and runs a full scan to return
        // nothing. `plan_match_query` and `plan_phrase_query` short-circuit the same way.
        if self.fragments.as_ref().is_some_and(Vec::is_empty) {
            return Ok(Arc::new(EmptyExec::new(fts_schema(
                DocumentGranularity::Row,
            ))));
        }

        // Kept per column; see `FlatStatsCoverage`.
        let mut per_column_coverage = Vec::with_capacity(query.column_names().len());
        let mut uncovered = RoaringBitmap::new();
        // Stale rows of every target column, by fragment id and row offset.
        let mut stale_addresses: HashMap<u32, RoaringBitmap> = HashMap::new();
        let mut any_indexed = false;
        let mut needs_whole_corpus = false;
        for column in query.column_names() {
            let column_uncovered: RoaringBitmap = match self
                .dataset
                .load_scalar_index(
                    IndexCriteria::default()
                        .for_column(column)
                        .supports_fts()
                        .with_fts_document_granularity(DocumentGranularity::Row),
                )
                .await?
            {
                Some(index) => {
                    any_indexed = true;
                    self.dataset
                        .unindexed_fragments(&index.name)
                        .await?
                        .iter()
                        .map(|fragment| fragment.id as u32)
                        .collect()
                }
                // No index on this column at all, so no fragment is covered for it.
                None => target_fragments
                    .iter()
                    .map(|fragment| fragment.id as u32)
                    .collect(),
            };

            // Rows this column's index does hold documents for, but whose entries a
            // newer data overlay made stale. A row's BM25F score needs every target
            // column's `tf_f`/`dl_f`, so a row stale in any column is scored wholly on
            // the flat side from its current values, and the indexed side blocks it.
            // The rows also join this column's fold, so a term the overlay introduced
            // reaches `docFreq_f`.
            let mut column_stale_rows = RowAddrTreeMap::new();
            match self
                .fts_overlay_plan(column, DocumentGranularity::Row, target_fragments)
                .await?
            {
                FtsOverlayPlan::Unchanged(_) => {}
                FtsOverlayPlan::RowLevel { stale_rows, .. } => {
                    column_stale_rows = self.stale_rows_in_id_domain(&stale_rows).await?;
                    for (fragment_id, offsets) in stale_rows {
                        *stale_addresses.entry(fragment_id).or_default() |= offsets;
                    }
                }
                // A legacy segment reports no fragment coverage, so no target fragment
                // can be proven free of stale entries and no row can be named as one.
                FtsOverlayPlan::FullScan => {
                    uncovered.extend(target_fragments.iter().map(|fragment| fragment.id as u32));
                    needs_whole_corpus = true;
                }
            }

            uncovered |= &column_uncovered;
            per_column_coverage.push(FlatStatsCoverage {
                fragments: column_uncovered,
                stale_rows: Arc::new(column_stale_rows),
            });
        }

        // BM25F needs one shared tokenizer configuration across the target columns
        // (`validate_combined_tokenizers`), and it is read off an index. Error out
        // when no target column has one rather than silently falling back to a
        // default tokenizer that may not match the data.
        if !any_indexed {
            return Err(Error::invalid_input(format!(
                "combined_fields requires an inverted index on at least one of {:?}",
                query.column_names().collect::<Vec<_>>()
            )));
        }

        // A stale row in a fragment the flat side reads whole is already read there.
        stale_addresses.retain(|fragment_id, _| !uncovered.contains(*fragment_id));
        let overlay_block = self.stale_rows_block_mask(&stale_addresses).await?;

        let uncovered_fragments: Vec<Fragment> = target_fragments
            .iter()
            .filter(|fragment| uncovered.contains(fragment.id as u32))
            .cloned()
            .collect();
        // The complement: fragments every target column indexes, and so the only
        // ones the indexed scan can score completely.
        let covered: RoaringBitmap = target_fragments
            .iter()
            .map(|fragment| fragment.id as u32)
            .filter(|fragment_id| !uncovered.contains(*fragment_id))
            .collect();

        // The only construction site for the indexed side, so the options every
        // shape needs cannot be set on one path and forgotten on another.
        //
        // `covered` restricts the scan to the fragments every target column indexes;
        // `None` means there is nothing to restrict. The overlay block keeps the
        // stale rows in those fragments out of the results. Segments stay
        // unrestricted either way: dropping a segment would drop its documents from
        // the corpus statistics too.
        //
        // `shared_scorer` is set only when a flat sibling exists: that side alone
        // sees the rows no index covers, so it publishes the corpus statistics both
        // children must score against.
        let index_exec = |covered: Option<RoaringBitmap>,
                          shared_scorer: Option<Arc<SharedFtsScorer<CombinedFieldsBM25Scorer>>>|
         -> Arc<dyn ExecutionPlan> {
            let mut exec = CombinedFieldsQueryExec::new(
                self.dataset.clone(),
                query.clone(),
                params.clone(),
                prefilter_source.clone(),
            )
            .with_external_mask(self.external_row_mask.clone());
            if let Some(covered) = covered {
                exec = exec.with_covered_fragments(covered);
            }
            if let Some(overlay_block) = overlay_block.clone() {
                exec = exec.with_overlay_block(overlay_block);
            }
            if let Some(shared_scorer) = shared_scorer {
                exec = exec.with_shared_scorer(shared_scorer);
            }
            Arc::new(exec)
        };

        // Every target column covers every target fragment, with no overlay-stale
        // entries anywhere: one unified scan that already emits merged hits sorted
        // by score with the top-k limit applied, so no union/sort is needed, and
        // there are no fragments to restrict the prefilter to.
        if uncovered_fragments.is_empty() && stale_addresses.is_empty() {
            return Ok(index_exec(None, None));
        }
        // `fast_search` is index-only by contract, but must still drop the
        // partially covered fragments and the stale rows so no partial or
        // pre-overlay score is emitted. When no
        // fragment is covered the answer is definitionally empty, and the index
        // exec would instead fail on the target column that has no segments.
        if self.fast_search {
            if covered.is_empty() {
                return Ok(Arc::new(EmptyExec::new(fts_schema(
                    DocumentGranularity::Row,
                ))));
            }
            return Ok(index_exec(Some(covered), None));
        }

        // A legacy segment names no stale rows, so nothing can be folded on top of the
        // index and the corpus has to be re-measured from every target fragment. That
        // is a full scan, so it stays confined to that case.
        let (flat_fragments, stats_scope) = if needs_whole_corpus {
            (target_fragments.to_vec(), FlatStatsScope::WholeCorpus)
        } else {
            (
                uncovered_fragments,
                FlatStatsScope::PerColumn(per_column_coverage),
            )
        };
        let mut scanned_fragments: RoaringBitmap = flat_fragments
            .iter()
            .map(|fragment| fragment.id as u32)
            .collect();
        scanned_fragments.extend(stale_addresses.keys().copied());
        // The flat side reads unfiltered and applies the filter only to the rows it
        // emits, as the indexed side does, so every row it reads counts toward the
        // corpus. Pushing the filter into the scan would make the corpus depend on
        // the filter, and on which rows an overlay routed onto this side.
        //
        // The emission prefilter is built over the fragments this scan reads, not
        // reused from the indexed child. `prefilter_source` spans the fragments the
        // target columns' indexes cover, so it names no row in an unindexed fragment,
        // and an allow-list can only be narrowed afterwards
        // (`build_prefilter_restricted_to_fragments`), never widened. Reusing it would
        // drop every match an unindexed fragment holds.
        let emit_prefilter = self
            .prefilter_source(filter_plan, scanned_fragments.clone())
            .await?;
        // Only a plan with both children needs the two to agree on a corpus, and a
        // whole-corpus scan leaves no fragment for an indexed child.
        let is_mixed = !needs_whole_corpus && flat_fragments.len() != target_fragments.len();
        let shared_scorer = is_mixed.then(|| Arc::new(SharedFtsScorer::new()));
        let flat_plan = self
            .plan_flat_combined_fields_query(
                flat_fragments,
                stale_addresses,
                scanned_fragments,
                stats_scope,
                emit_prefilter,
                query,
                params,
                shared_scorer.clone(),
            )
            .await?;
        // The flat exec emits in scan order and applies no limit, so even with no
        // index child the plan still needs the score sort and top-k fetch.
        let scored_plan = if !is_mixed {
            flat_plan
        } else {
            let union =
                UnionExec::try_new(vec![index_exec(Some(covered), shared_scorer), flat_plan])?;
            Arc::new(RepartitionExec::try_new(
                union,
                Partitioning::RoundRobinBatch(1),
            )?)
        };
        Ok(Arc::new(
            SortExec::new(
                Self::fts_score_sort_exprs(scored_plan.schema().as_ref())?.into(),
                scored_plan,
            )
            .with_fetch(params.limit),
        ))
    }

    /// Plan the flat (unindexed) side of a `combined_fields` query: one scan over
    /// `fragments` and the `stale_rows` taken by address, projecting every target
    /// column, scored as one virtual field. `scanned_fragments` names every
    /// fragment either input reads from.
    ///
    /// The scan is unfiltered; `emit_prefilter` picks the rows it emits.
    ///
    /// Emits in scan order with no limit applied; the caller supplies the score
    /// sort and top-k fetch.
    #[allow(clippy::too_many_arguments)]
    async fn plan_flat_combined_fields_query(
        &self,
        fragments: Vec<Fragment>,
        stale_rows: HashMap<u32, RoaringBitmap>,
        scanned_fragments: RoaringBitmap,
        stats_scope: FlatStatsScope,
        emit_prefilter: PreFilterSource,
        query: &CombinedFieldsQuery,
        params: &FtsSearchParams,
        shared_scorer: Option<Arc<SharedFtsScorer<CombinedFieldsBM25Scorer>>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // A path that continues past a `List` is not addressable by a projection,
        // so such a column is scanned through its list root and walked down to the
        // leaf by `FlatCombinedFieldsExec`, exactly as `plan_flat_match_query`
        // does for a single column.
        let resolved_fields = query
            .column_names()
            .map(|column| {
                resolve_fts_field(self.dataset.schema(), column, DocumentGranularity::Row)
            })
            .collect::<Result<Vec<_>>>()?;
        let columns = resolved_fields
            .iter()
            .map(|resolved| {
                if resolved.has_lists() {
                    resolved.root_column.clone()
                } else {
                    resolved.canonical_path.clone()
                }
            })
            .collect::<Vec<_>>();
        let mut plan = self
            .plan_uncovered_rows_scan(fragments, stale_rows, columns, &ExprFilterPlan::default())
            .await?;
        // A nested target column needs a flat alias so the exec can resolve it by
        // name in the batch schema. A list-bearing one gets its flat column from
        // the exec instead, since no projection can produce it.
        for (column, resolved) in query.column_names().zip(&resolved_fields) {
            if !resolved.has_lists() {
                plan = self.ensure_column_alias(plan, column)?;
            }
        }

        let mut flat_plan = FlatCombinedFieldsExec::try_new(
            self.dataset.clone(),
            query.clone(),
            params.clone(),
            plan,
            scanned_fragments,
            resolved_fields,
            stats_scope,
            emit_prefilter,
        )?;
        if let Some(shared_scorer) = shared_scorer {
            flat_plan = flat_plan.with_shared_scorer(shared_scorer);
        }
        let flat_plan: Arc<dyn ExecutionPlan> = Arc::new(flat_plan);
        // The rows this side scores never reach an index-side prefilter, so the
        // external row-address mask is applied to its output here, as the flat
        // MatchQuery branch does. It restricts what is emitted, not the corpus the
        // scores are computed against, so both children of a mixed plan keep
        // scoring against the same statistics.
        if let Some(mask) = self.external_row_mask.clone() {
            return Ok(Arc::new(RowAddrMaskFilterExec::new(flat_plan, mask)));
        }
        Ok(flat_plan)
    }

    /// Plan FTS overlay handling at row granularity.
    ///
    /// Modern segments remain searchable while their overlay-stale rows are blocked and
    /// re-evaluated from current values. A legacy segment without fragment coverage falls back
    /// to a full target scan when a relevant overlay exists because its indexed row set is
    /// unknown.
    async fn fts_overlay_plan(
        &self,
        column: &str,
        document_granularity: DocumentGranularity,
        target_fragments: &[Fragment],
    ) -> Result<FtsOverlayPlan> {
        if target_fragments.iter().all(|f| f.overlays.is_empty()) {
            return Ok(FtsOverlayPlan::Unchanged(None));
        }

        let Some(segments) = load_segments(&self.dataset, column, document_granularity).await?
        else {
            return Ok(FtsOverlayPlan::Unchanged(None));
        };

        let overlaid_frags = overlaid_fragments(target_fragments);
        let mut stale_rows = HashMap::new();
        for segment in &segments {
            if segment.fragment_bitmap.is_none() {
                let mut legacy_stale_rows = HashMap::new();
                collect_overlay_stale_rows_for_segment(
                    segment,
                    &overlaid_frags,
                    &mut legacy_stale_rows,
                    self.dataset.schema(),
                )?;
                if !legacy_stale_rows.is_empty() {
                    return Ok(FtsOverlayPlan::FullScan);
                }
            } else {
                collect_overlay_stale_rows_for_segment(
                    segment,
                    &overlaid_frags,
                    &mut stale_rows,
                    self.dataset.schema(),
                )?;
            }
        }

        if stale_rows.is_empty() {
            Ok(FtsOverlayPlan::Unchanged(Some(segments)))
        } else {
            Ok(FtsOverlayPlan::RowLevel {
                stale_rows,
                segments,
            })
        }
    }

    /// Here we use a full text search as a post-filter. Rows are retained
    /// according to the match query's token operator.
    ///
    /// Only valid (currently) for match queries.
    pub(super) async fn flat_fts_filter(
        &self,
        input: Arc<dyn ExecutionPlan>,
        q: &FullTextSearchQuery,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let fts_query = &q.query;

        match fts_query {
            FtsQuery::Match(match_query) => {
                let schema = Arc::new((input.schema()).try_with_column(SCORE_FIELD.clone())?);

                let column = match_query
                    .column
                    .as_ref()
                    .ok_or(Error::invalid_input(
                        "the column must be specified in the query".to_string(),
                    ))?
                    .clone();
                let document_granularity = match_query.document_granularity.ok_or_else(|| {
                    Error::internal("FTS Match query granularity was not resolved".to_string())
                })?;
                let resolved =
                    resolve_fts_field(self.dataset.schema(), &column, document_granularity)?;
                let scan_column = if resolved.has_lists() {
                    resolved.root_column.clone()
                } else {
                    resolved.canonical_path.clone()
                };
                let input = if schema.column_with_name(&scan_column).is_none() {
                    let projection = self
                        .dataset
                        .empty_projection()
                        .union_column(&scan_column, OnMissing::Error)?;
                    let input = self.take(input, projection)?;
                    if resolved.has_lists() {
                        input
                    } else {
                        self.ensure_column_alias(input, &scan_column)?
                    }
                } else {
                    input
                };

                Ok(Arc::new(FlatMatchFilterExec::new_with_resolved_field(
                    input,
                    self.dataset.clone(),
                    match_query.clone(),
                    q.params(),
                    resolved,
                )))
            }
            _ => Err(Error::not_supported(
                "Only Match queries are supported currently when using FTS as a post-filter",
            )),
        }
    }

    /// Here we consume all input (as unindexed) and rerank according to BM25 scores
    ///
    /// If there is an index on the column then we still use the index to determine the
    /// tokenizer and inform the BM25 scoring (e.g. avg doc length, token frequency, etc.)
    async fn fts_rerank(
        &self,
        input: Arc<dyn ExecutionPlan>,
        q: &FullTextSearchQuery,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let fts_query = &q.query;

        match fts_query {
            FtsQuery::Match(match_query) => {
                let schema = Arc::new((input.schema()).try_with_column(SCORE_FIELD.clone())?);

                let column = match_query
                    .column
                    .as_ref()
                    .ok_or(Error::invalid_input(
                        "the column must be specified in the query".to_string(),
                    ))?
                    .clone();
                let document_granularity = match_query.document_granularity.ok_or_else(|| {
                    Error::internal("FTS Match query granularity was not resolved".to_string())
                })?;
                let resolved =
                    resolve_fts_field(self.dataset.schema(), &column, document_granularity)?;
                let scan_column = if resolved.has_lists() {
                    resolved.root_column.clone()
                } else {
                    resolved.canonical_path.clone()
                };
                let document_column = if resolved.has_lists() {
                    VALUE_COLUMN_NAME.to_string()
                } else {
                    resolved.canonical_path.clone()
                };
                let input = if schema.column_with_name(&scan_column).is_none() {
                    let projection = self
                        .dataset
                        .empty_projection()
                        .union_column(&scan_column, OnMissing::Error)?;
                    let input = self.take(input, projection)?;
                    if resolved.has_lists() {
                        input
                    } else {
                        self.ensure_column_alias(input, &document_column)?
                    }
                } else {
                    input
                };
                let input = if resolved.has_lists() {
                    Arc::new(FtsDocumentExec::new(input, resolved)) as Arc<dyn ExecutionPlan>
                } else {
                    input
                };

                Ok(Arc::new(FlatMatchQueryExec::new_with_document_granularity(
                    self.dataset.clone(),
                    match_query.clone(),
                    q.params(),
                    input,
                    document_granularity,
                    document_column,
                )))
            }
            _ => {
                let default_filter = ExprFilterPlan::default();
                let fts_plan = self.fts(&default_filter, q).await?;

                let vector_row_id = Column::new_with_schema(ROW_ID, input.schema().as_ref())?;
                let fts_row_id = Column::new_with_schema(ROW_ID, fts_plan.schema().as_ref())?;
                let join = HashJoinExec::try_new(
                    input,
                    fts_plan,
                    vec![(Arc::new(vector_row_id), Arc::new(fts_row_id))],
                    None,
                    &JoinType::Inner,
                    None,
                    PartitionMode::CollectLeft,
                    NullEquality::NullEqualsNull,
                    false,
                )?;

                let schema = join.schema();
                let mut projection_exprs = Vec::new();
                let mut contain_rowid = false;
                for field in schema.fields() {
                    if field.name() == ROW_ID {
                        if contain_rowid {
                            continue;
                        }
                        contain_rowid = true;
                    }
                    projection_exprs.push((
                        Arc::new(Column::new_with_schema(field.name(), schema.as_ref())?)
                            as Arc<dyn PhysicalExpr>,
                        field.name().clone(),
                    ));
                }

                let projection_exec = ProjectionExec::try_new(projection_exprs, Arc::new(join))?;
                Ok(Arc::new(projection_exec))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use lance_index::scalar::inverted::query::{
        BooleanQuery, BoostQuery, FtsQuery, MatchQuery, MultiMatchQuery, Occur, PhraseQuery,
    };

    #[test]
    fn test_fts_query_contract_rejects_invalid_values() {
        let negative_match = FtsQuery::Match(MatchQuery::new("hello".to_string()).with_boost(-1.0));
        let error = validate_fts_query_contract(&negative_match).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("finite and non-negative"));

        let empty_boolean = FtsQuery::Boolean(BooleanQuery::new(Vec::<(Occur, FtsQuery)>::new()));
        let error = validate_fts_query_contract(&empty_boolean).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("at least one should/must query"));

        let infinite_boost = FtsQuery::Boost(BoostQuery::new(
            MatchQuery::new("hello".to_string()).into(),
            MatchQuery::new("world".to_string()).into(),
            Some(f32::INFINITY),
        ));
        let error = validate_fts_query_contract(&infinite_boost).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("BoostQuery negative_boost"));
    }

    #[test]
    fn test_query_local_residual_row_bound() {
        let fragment_with_rows = |id, physical_rows| {
            let mut fragment = Fragment::new(id);
            fragment.physical_rows = physical_rows;
            fragment
        };

        assert!(has_bounded_query_local_residual_rows(&[
            fragment_with_rows(0, Some(40_000)),
            fragment_with_rows(1, Some(60_000)),
        ]));
        assert!(!has_bounded_query_local_residual_rows(&[
            fragment_with_rows(0, Some(40_000)),
            fragment_with_rows(1, Some(60_001)),
        ]));
        assert!(!has_bounded_query_local_residual_rows(&[
            fragment_with_rows(0, Some(1)),
            fragment_with_rows(1, None),
        ]));
    }

    #[test]
    fn test_normalize_fts_zero_boosts_recurses_and_preserves_nonzero_values() {
        fn boost_bits(query: &FtsQuery) -> Vec<u32> {
            match query {
                FtsQuery::Match(query) => vec![query.boost.to_bits()],
                FtsQuery::Phrase(_) | FtsQuery::CombinedFields(_) => Vec::new(),
                FtsQuery::Boost(query) => std::iter::once(query.negative_boost.to_bits())
                    .chain(boost_bits(&query.positive))
                    .chain(boost_bits(&query.negative))
                    .collect(),
                FtsQuery::MultiMatch(query) => query
                    .match_queries
                    .iter()
                    .map(|query| query.boost.to_bits())
                    .collect(),
                FtsQuery::Boolean(query) => query
                    .should
                    .iter()
                    .chain(&query.must)
                    .chain(&query.must_not)
                    .flat_map(boost_bits)
                    .collect(),
            }
        }

        let match_query =
            |terms: &str, boost| MatchQuery::new(terms.to_string()).with_boost(boost).into();
        let multi_match = MultiMatchQuery::try_new(
            "needle".to_string(),
            vec!["title".to_string(), "body".to_string()],
        )
        .unwrap()
        .try_with_boosts(vec![-0.0, 2.5])
        .unwrap();
        let negative = BooleanQuery::new([
            (Occur::Should, multi_match.into()),
            (Occur::Must, match_query("required", 3.5)),
            (Occur::MustNot, match_query("blocked", -0.0)),
        ]);
        let boost = BoostQuery::new(match_query("positive", -0.0), negative.into(), Some(-0.0));
        let mut query: FtsQuery = BooleanQuery::new([
            (Occur::Should, match_query("outer", -0.0)),
            (Occur::Must, boost.into()),
            (Occur::MustNot, match_query("unchanged", 4.5)),
        ])
        .into();

        let nz = (-0.0_f32).to_bits();
        let pz = 0.0_f32.to_bits();
        let b2 = 2.5_f32.to_bits();
        let b3 = 3.5_f32.to_bits();
        let b4 = 4.5_f32.to_bits();
        assert_eq!(boost_bits(&query), vec![nz, nz, nz, nz, b2, b3, nz, b4]);
        normalize_fts_zero_boosts(&mut query);
        assert_eq!(boost_bits(&query), vec![pz, pz, pz, pz, b2, b3, pz, b4]);
    }

    #[test]
    fn test_dataset_planner_defers_auto_fuzziness_recursively() {
        fn collect_fuzziness(query: &FtsQuery, values: &mut Vec<Option<u32>>) {
            match query {
                FtsQuery::Match(query) => values.push(query.fuzziness),
                FtsQuery::Phrase(_) | FtsQuery::CombinedFields(_) => {}
                FtsQuery::Boost(query) => {
                    collect_fuzziness(&query.positive, values);
                    collect_fuzziness(&query.negative, values);
                }
                FtsQuery::MultiMatch(query) => {
                    values.extend(query.match_queries.iter().map(|query| query.fuzziness));
                }
                FtsQuery::Boolean(query) => {
                    for child in query
                        .should
                        .iter()
                        .chain(&query.must)
                        .chain(&query.must_not)
                    {
                        collect_fuzziness(child, values);
                    }
                }
            }
        }

        let auto_match = |terms: &str| {
            MatchQuery::new(terms.to_owned())
                .with_fuzziness(None)
                .into()
        };
        let mut multi_match = MultiMatchQuery::try_new(
            "multi".to_owned(),
            vec!["title".to_owned(), "body".to_owned()],
        )
        .unwrap();
        multi_match.match_queries[0].fuzziness = None;
        multi_match.match_queries[1].fuzziness = Some(1);
        let boost = BoostQuery::new(
            auto_match("positive"),
            MatchQuery::new("negative".to_owned())
                .with_fuzziness(Some(0))
                .into(),
            None,
        );
        let mut query: FtsQuery = BooleanQuery::new([
            (Occur::Should, auto_match("root")),
            (Occur::Must, FtsQuery::MultiMatch(multi_match)),
            (Occur::MustNot, boost.into()),
        ])
        .into();

        apply_dataset_planner_auto_fuzziness_compatibility_gate(&mut query);

        let mut fuzziness = Vec::new();
        collect_fuzziness(&query, &mut fuzziness);
        assert_eq!(
            fuzziness,
            [Some(0), Some(0), Some(1), Some(0), Some(0)],
            "AUTO must become exact without changing explicit fuzzy or exact leaves"
        );
    }

    #[test]
    fn test_compound_scorer_shape_supports_cross_column_boolean_queries() {
        let query = FtsQuery::Boolean(BooleanQuery::new([
            (
                Occur::Should,
                MatchQuery::new("alpha".to_string())
                    .with_column(Some("title".to_string()))
                    .into(),
            ),
            (
                Occur::Must,
                MatchQuery::new("beta".to_string())
                    .with_column(Some("body".to_string()))
                    .into(),
            ),
            (
                Occur::MustNot,
                MatchQuery::new("gamma".to_string())
                    .with_column(Some("summary".to_string()))
                    .into(),
            ),
        ]));

        assert!(supports_compound_scorer(&query));
        assert_eq!(
            collect_fts_columns_in_order(&query),
            ["title", "body", "summary"]
        );
    }

    #[test]
    fn test_compound_scorer_leaves_top_level_cross_column_multi_match_on_existing_path() {
        let single_column = FtsQuery::MultiMatch(
            MultiMatchQuery::try_new("alpha".to_string(), vec!["title".to_string()]).unwrap(),
        );
        let cross_column = FtsQuery::MultiMatch(
            MultiMatchQuery::try_new(
                "alpha".to_string(),
                vec!["title".to_string(), "body".to_string()],
            )
            .unwrap(),
        );

        assert!(supports_compound_scorer(&single_column));
        assert!(!supports_compound_scorer(&cross_column));
    }

    #[test]
    fn test_collect_phrase_columns_traverses_prohibited_subtrees() {
        let phrase =
            PhraseQuery::new("exact phrase".to_string()).with_column(Some("body".to_string()));
        let query = FtsQuery::Boolean(BooleanQuery::new([
            (
                Occur::Must,
                MatchQuery::new("alpha".to_string())
                    .with_column(Some("title".to_string()))
                    .into(),
            ),
            (Occur::MustNot, phrase.into()),
        ]));
        let mut columns = HashSet::new();

        collect_phrase_columns(&query, &mut columns);

        assert_eq!(columns, HashSet::from(["body".to_string()]));
    }
}
