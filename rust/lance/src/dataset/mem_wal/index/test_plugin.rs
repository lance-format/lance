// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! A built-in memtable index plugin under another type with the same name, for
//! testing what the memtable assumes about plugins it does not know.

use std::sync::Arc;

use arrow_array::RecordBatch;
use datafusion::common::ScalarValue;
use lance_core::{Error, Result};
use lance_index::IndexType;
use lance_index::scalar::ScalarIndexParams;
use lance_index::scalar::expression::ScalarQueryParser;
use lance_index::scalar::registry::TrainingCriteria;

use super::fts::FtsQueryExpr;
use super::{
    FlushContext, FlushOutcome, FtsMemQuery, MemIndex, MemIndexBuildContext, MemIndexPlugin,
    MemIndexSpec, MemMatches, MemQuery, MemSearchResult, PositionSet, PrimaryKeyIndex, RankedMatch,
    ResolveContext, ResolvedIndex, RowPosition, SearchContext, VectorMemQuery,
};

/// How a wrapped index departs from the one it wraps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Deviation {
    /// None: only the type differs.
    None,
    /// Lists every match, whatever budget it is offered.
    IgnoresMatchBudget,
    /// Answers probes, declines every real search.
    DeclinesRealSearches,
    /// Declines probes, answers every real search.
    DeclinesProbes,
    /// Accepts every query while planning, then declines it; a filter, which
    /// planning does not ask about, is simply declined.
    AcceptsThenDeclines,
    /// Accepts every query while planning, then fails it.
    AcceptsThenFails,
    /// Answers a filter with every visible row as a candidate.
    AnswersCandidates,
    /// Hands out a new primary-key capability each time it is asked.
    FreshKeyCapability,
    /// Asks the flush to build it from the generation.
    AsksForABuild,
    /// Adds the first row past the visible ones to every answer, ranked first.
    AnswersPastVisible,
    /// Covers the key column but cannot serve as the primary-key index.
    NotAKeyIndex,
    /// Fails its flush.
    FailsToFlush,
    /// Records the file it wrote under another index name.
    RecordsAnotherName,
    /// Panics on every insert.
    PanicsOnInsert,
    /// Rejects its flush parameters.
    RejectsFlushParams,
}

/// `spec`, maintained by its plugin wrapped with `deviation`.
pub fn wrapped(spec: MemIndexSpec, deviation: Deviation) -> MemIndexSpec {
    wrapped_with_flush_params(spec, deviation, None)
}

/// As [`wrapped`], building on disk with `flush_params` when given.
pub fn wrapped_with_flush_params(
    spec: MemIndexSpec,
    deviation: Deviation,
    flush_params: Option<ScalarIndexParams>,
) -> MemIndexSpec {
    MemIndexSpec {
        plugin: Arc::new(Wrapped {
            inner: spec.plugin.clone(),
            deviation,
            flush_params,
        }),
        ..spec
    }
}

#[derive(Debug)]
struct Wrapped {
    inner: Arc<dyn MemIndexPlugin>,
    deviation: Deviation,
    flush_params: Option<ScalarIndexParams>,
}

#[async_trait::async_trait]
impl MemIndexPlugin for Wrapped {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn details_message(&self) -> &str {
        self.inner.details_message()
    }
    fn flush_index_type(&self) -> IndexType {
        self.inner.flush_index_type()
    }
    fn training_criteria(&self) -> TrainingCriteria {
        self.inner.training_criteria()
    }
    fn flush_params(&self, spec: &MemIndexSpec) -> Result<ScalarIndexParams> {
        if self.deviation == Deviation::RejectsFlushParams {
            return Err(Error::invalid_input("flush parameters rejected"));
        }
        match &self.flush_params {
            Some(params) => Ok(params.clone()),
            None => self.inner.flush_params(spec),
        }
    }
    fn query_parser(
        &self,
        index_name: String,
        index_details: Option<&prost_types::Any>,
    ) -> Option<Box<dyn ScalarQueryParser>> {
        self.inner.query_parser(index_name, index_details)
    }
    async fn resolve(&self, ctx: &ResolveContext<'_>) -> Result<ResolvedIndex> {
        self.inner.resolve(ctx).await
    }
    fn validate(&self, ctx: &MemIndexBuildContext<'_>) -> Result<()> {
        self.inner.validate(ctx)
    }
    fn create(&self, ctx: &MemIndexBuildContext<'_>) -> Result<Arc<dyn MemIndex>> {
        Ok(Arc::new(WrappedIndex {
            inner: self.inner.create(ctx)?,
            deviation: self.deviation,
        }))
    }
}

#[derive(Debug)]
struct WrappedIndex {
    inner: Arc<dyn MemIndex>,
    deviation: Deviation,
}

/// A probe rather than a real search: a vector search for no neighbours, or a
/// full-text match on no text.
fn is_probe(query: &dyn MemQuery) -> bool {
    let any = query.as_any();
    any.downcast_ref::<VectorMemQuery>()
        .is_some_and(|query| query.k == 0)
        || any.downcast_ref::<FtsMemQuery>().is_some_and(
            |query| matches!(&query.expr, FtsQueryExpr::Match { query, .. } if query.is_empty()),
        )
}

#[async_trait::async_trait]
impl MemIndex for WrappedIndex {
    fn columns(&self) -> &[String] {
        self.inner.columns()
    }
    fn can_answer(&self, query: &dyn MemQuery) -> bool {
        self.inner.can_answer(query)
            && match self.deviation {
                Deviation::DeclinesRealSearches => is_probe(query),
                Deviation::DeclinesProbes => !is_probe(query),
                _ => true,
            }
    }
    fn insert(&self, batch: &RecordBatch, row_offset: RowPosition) -> Result<()> {
        assert!(
            self.deviation != Deviation::PanicsOnInsert,
            "insert panicked"
        );
        self.inner.insert(batch, row_offset)
    }
    fn resident_bytes(&self) -> usize {
        self.inner.resident_bytes()
    }
    fn search(&self, query: &dyn MemQuery, ctx: &SearchContext) -> Result<Option<MemMatches>> {
        match self.deviation {
            Deviation::AcceptsThenDeclines => Ok(None),
            Deviation::AcceptsThenFails => Err(lance_core::Error::internal("search failed")),
            Deviation::AnswersCandidates => Ok(Some(MemMatches::Filter(MemSearchResult::at_most(
                PositionSet::all_visible(ctx.max_visible),
            )))),
            Deviation::AnswersPastVisible => {
                let hidden = ctx.max_visible + 1;
                Ok(match self.inner.search(query, ctx)? {
                    Some(MemMatches::Filter(mut result)) => {
                        result.at_least.insert(hidden);
                        result.at_most.insert(hidden);
                        Some(MemMatches::Filter(result))
                    }
                    Some(MemMatches::Ranked(mut matches)) => {
                        matches.insert(0, RankedMatch::new(hidden, 0.0));
                        Some(MemMatches::Ranked(matches))
                    }
                    None => None,
                })
            }
            _ if !self.can_answer(query) => Ok(None),
            Deviation::IgnoresMatchBudget => self
                .inner
                .search(query, &SearchContext::new(ctx.max_visible)),
            _ => self.inner.search(query, ctx),
        }
    }
    async fn flush(&self, ctx: &FlushContext<'_>) -> Result<FlushOutcome> {
        match self.deviation {
            Deviation::AsksForABuild => Ok(FlushOutcome::BuildFromGeneration),
            Deviation::FailsToFlush => Err(Error::internal("flush failed")),
            Deviation::RecordsAnotherName => match self.inner.flush(ctx).await? {
                FlushOutcome::Wrote(mut index_meta) => {
                    index_meta.name = "another_name".to_string();
                    Ok(FlushOutcome::Wrote(index_meta))
                }
                outcome => Ok(outcome),
            },
            _ => self.inner.flush(ctx).await,
        }
    }
    fn as_primary_key(self: Arc<Self>) -> Option<Arc<dyn PrimaryKeyIndex>> {
        let key = self.inner.clone().as_primary_key()?;
        match self.deviation {
            Deviation::FreshKeyCapability => Some(Arc::new(KeyCapability(key))),
            Deviation::NotAKeyIndex => None,
            _ => Some(key),
        }
    }
}

/// A primary-key capability allocated afresh, around the same index.
struct KeyCapability(Arc<dyn PrimaryKeyIndex>);

impl PrimaryKeyIndex for KeyCapability {
    fn insert_and_report_existing(
        &self,
        batch: &RecordBatch,
        row_offset: RowPosition,
    ) -> Result<bool> {
        self.0.insert_and_report_existing(batch, row_offset)
    }
    fn newest_visible(&self, key: &ScalarValue, max_visible: RowPosition) -> Option<RowPosition> {
        self.0.newest_visible(key, max_visible)
    }
    fn training_batches(&self, batch_size: usize) -> Result<Vec<RecordBatch>> {
        self.0.training_batches(batch_size)
    }
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}
