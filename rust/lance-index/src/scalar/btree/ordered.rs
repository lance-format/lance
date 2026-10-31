// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Ordered search over a single-column B-tree.
//!
//! Pages store keys ascending with nulls first, and the lookup lists pages in
//! that order, so the null run sits at the front. A walk splits the candidate
//! pages into a null run and a value run, reverses both for a descending
//! order, and puts the null run first or last:
//!
//! ```text
//! lookup:            [N N] [N 1 2] [3 4 5]
//!
//! ASC  NULLS FIRST:  [N N] [N 1 2] [3 4 5]
//! ASC  NULLS LAST:   [1 2] [3 4 5] [N N] [N]
//! DESC NULLS LAST:   [5 4 3] [2 1 N] [N N]
//! DESC NULLS FIRST:  [N] [N N] [5 4 3] [2 1]
//! ```
//!
//! A page holding both null and non-null keys is read once in ASC NULLS FIRST
//! and DESC NULLS LAST, where it ends one run and starts the next. In the other
//! two orders it sits at both ends of the walk, so the walk reads it twice,
//! once for each run. The walk sorts each page it reads by key, then by
//! ascending id, so equal keys within one page come in ascending id order in
//! both directions. Training sorts by key alone, so equal keys on different
//! pages have no id order.

use std::cmp::Ordering;
use std::iter;
use std::ops::{Bound, Range};
use std::sync::{Arc, OnceLock};

use arrow::compute::{and, is_not_null, is_null};
use arrow_array::{ArrayRef, RecordBatch, cast::AsArray};
use arrow_ord::ord::make_comparator;
use arrow_ord::sort::{SortColumn, lexsort_to_indices};
use arrow_schema::{DataType, Field, Schema, SchemaRef, SortOptions};
use arrow_select::{filter::filter_record_batch, take::take_record_batch};
use datafusion_common::{DataFusionError, ScalarValue};
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_plan::{SendableRecordBatchStream, stream::RecordBatchStreamAdapter};
use futures::{StreamExt, TryStreamExt, future, stream};
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::utils::tracing::{IO_TYPE_LOAD_SCALAR_PART, TRACE_IO_EVENTS};
use lance_core::{Error, ROW_ADDR, ROW_ID, Result};
use lance_index_core::remapping::remap_record_batch_async;
use lance_index_core::scalar::OrderedScalarIndex;
use tracing::info;

use super::{
    BTreeIndex, BTreeLookup, LazyIndexReader, Matches, OrderableScalarValue, partition_point,
};
use crate::metrics::MetricsCollector;
use crate::scalar::registry::VALUE_COLUMN_NAME;
use crate::scalar::{AnyQuery, SargableQuery, compute_next_prefix};

const READ_AHEAD_PAGES: usize = 2;

const STORED_ORDER: SortOptions = SortOptions {
    descending: false,
    nulls_first: true,
};

#[derive(Clone)]
enum RowSelection {
    All,
    Nulls,
    Values(Option<Arc<dyn PhysicalExpr>>),
}

struct PageVisit {
    page: u32,
    rows: RowSelection,
}

type Visits = Box<dyn DoubleEndedIterator<Item = PageVisit> + Send>;

/// A lookup that passed [`check_page_order`]. Each page's min is at most its
/// max, each max is at most the next page's min, and page ids ascend with
/// lookup rows.
#[derive(Clone, Copy)]
struct CheckedLookup<'a>(&'a Arc<BTreeLookup>);

enum ValuePages {
    Rows {
        rows: Range<usize>,
        covered: Range<usize>,
    },
    Matched(Vec<Matches>),
}

impl OrderedScalarIndex for BTreeIndex {
    fn search_ordered(
        &self,
        query: Option<&dyn AnyQuery>,
        options: SortOptions,
        metrics: Arc<dyn MetricsCollector>,
    ) -> Result<SendableRecordBatchStream> {
        let lookup = check_page_order(&self.page_lookup, metrics.as_ref())?;
        let mut walk = self.plan_walk(lookup, query, options)?;
        let id_column = if self.results_are_row_addresses {
            ROW_ADDR
        } else {
            ROW_ID
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new(VALUE_COLUMN_NAME, self.data_type.clone(), true),
            Field::new(id_column, DataType::UInt64, false),
        ]));
        let index = Arc::new(self.clone());
        let reader = LazyIndexReader::new(self.store.clone(), self.ranges_to_files.clone());
        let batch_schema = schema.clone();
        let read = move |visit| {
            let index = index.clone();
            let reader = reader.clone();
            let schema = batch_schema.clone();
            let metrics = metrics.clone();
            async move {
                index
                    .read_visit(visit, &reader, options, schema, metrics.as_ref())
                    .await
            }
        };
        // Read-ahead starts after the first page, so a consumer that stops
        // there pays for one page read.
        let first = stream::iter(walk.next()).then(read.clone());
        let rest = stream::iter(walk).map(read).buffered(1 + READ_AHEAD_PAGES);
        let batches = first
            .chain(rest)
            .try_filter(|batch| future::ready(batch.num_rows() > 0))
            .map_err(DataFusionError::from);
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, batches)))
    }
}

impl BTreeIndex {
    fn plan_walk(
        &self,
        lookup: CheckedLookup<'_>,
        query: Option<&dyn AnyQuery>,
        options: SortOptions,
    ) -> Result<Visits> {
        let query = query
            .map(|query| {
                let query = query
                    .as_any()
                    .downcast_ref::<SargableQuery>()
                    .ok_or_else(|| {
                        Error::invalid_input(format!(
                            "B-tree ordered search expects a SargableQuery, got {query:?}"
                        ))
                    })?;
                strings_in_key_type(query, &self.data_type)
            })
            .transpose()?;
        let (mut null_run, value_pages, predicate) = match &query {
            // An unbounded range is the literal `true`, so it selects null keys too.
            None | Some(SargableQuery::Range(Bound::Unbounded, Bound::Unbounded)) => (
                null_pages(lookup),
                ValuePages::between(lookup, Bound::Unbounded, Bound::Unbounded)?,
                None,
            ),
            Some(query) => {
                let null_run: Visits = if matches!(query, SargableQuery::IsNull()) {
                    null_pages(lookup)
                } else {
                    Box::new(iter::empty())
                };
                let value_pages = Self::value_pages(lookup, query)?;
                let predicate = if value_pages.is_empty() {
                    None
                } else {
                    Some(self.compile_predicate(query)?)
                };
                (null_run, value_pages, predicate)
            }
        };

        // Indices written before Lance 0.25.2 record a page's min without its
        // nulls, so even a fully covered page can hold null keys.
        let mut value_run = value_pages.visits(
            lookup,
            RowSelection::Values(None),
            RowSelection::Values(predicate),
        )?;
        if options.descending {
            null_run = Box::new(null_run.rev());
            value_run = Box::new(value_run.rev());
        }
        let (mut walk, mut rest) = if options.nulls_first {
            (null_run, value_run)
        } else {
            (value_run, null_run)
        };
        let (last, next) = (walk.next_back(), rest.next());
        let shared_page = match (&last, &next) {
            (Some(last), Some(next)) if last.page == next.page => last.rows.union(&next.rows),
            _ => None,
        };
        let junction = match shared_page {
            Some(rows) => [next.map(|next| PageVisit { rows, ..next }), None],
            None => [last, next],
        };
        Ok(Box::new(
            walk.chain(junction.into_iter().flatten()).chain(rest),
        ))
    }

    fn value_pages(lookup: CheckedLookup<'_>, query: &SargableQuery) -> Result<ValuePages> {
        match query {
            SargableQuery::Equals(value) if !value.is_null() => lookup
                .0
                .pages_eq(&OrderableScalarValue(value.clone()))
                .map(ValuePages::Matched),
            SargableQuery::Range(start, end) if !is_null_bound(start) && !is_null_bound(end) => {
                ValuePages::between(lookup, start.as_ref(), end.as_ref())
            }
            SargableQuery::IsIn(values) => lookup
                .0
                .pages_in(
                    values
                        .iter()
                        .filter(|value| !value.is_null())
                        .cloned()
                        .map(OrderableScalarValue),
                )
                .map(ValuePages::Matched),
            SargableQuery::LikePrefix(
                prefix @ (ScalarValue::Utf8(Some(text))
                | ScalarValue::LargeUtf8(Some(text))
                | ScalarValue::Utf8View(Some(text))),
            ) => {
                let key = |text: String| match prefix {
                    ScalarValue::LargeUtf8(_) => ScalarValue::LargeUtf8(Some(text)),
                    _ => ScalarValue::Utf8(Some(text)),
                };
                let start = key(text.clone());
                let end = compute_next_prefix(text).map(key);
                ValuePages::between(
                    lookup,
                    Bound::Included(&start),
                    end.as_ref().map_or(Bound::Unbounded, Bound::Excluded),
                )
            }
            SargableQuery::LikePrefix(prefix)
                if !matches!(
                    prefix.data_type(),
                    DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
                ) =>
            {
                Err(Error::invalid_input(format!(
                    "B-tree ordered search needs a string LIKE prefix, got {prefix} of type {}",
                    prefix.data_type()
                )))
            }
            SargableQuery::FullTextSearch(_) => Err(Error::invalid_input(
                "full text search is not supported for BTree index, build an inverted index for it",
            )),
            SargableQuery::Equals(_)
            | SargableQuery::Range(..)
            | SargableQuery::LikePrefix(_)
            | SargableQuery::IsNull() => Ok(ValuePages::Matched(Vec::new())),
        }
    }

    async fn read_visit(
        &self,
        visit: PageVisit,
        reader: &LazyIndexReader,
        options: SortOptions,
        schema: SchemaRef,
        metrics: &dyn MetricsCollector,
    ) -> Result<RecordBatch> {
        let page = reader
            .get()
            .await?
            .read_record_batch(visit.page as u64, self.batch_size)
            .await?;
        metrics.record_single_page_read();
        metrics.record_part_load();
        info!(target: TRACE_IO_EVENTS, r#type=IO_TYPE_LOAD_SCALAR_PART, index_type="btree", part_id=visit.page);
        let page = match (&self.batch_remapper, &self.frag_reuse_index) {
            (Some(remapper), _) => remap_record_batch_async(remapper.as_ref(), page, 1).await?,
            (None, Some(frag_reuse_index)) => {
                frag_reuse_index.remap_row_ids_record_batch(page, 1)?
            }
            (None, None) => page,
        };
        let page = visit.rows.select(page)?;
        let sorted = lexsort_to_indices(
            &[
                SortColumn {
                    values: page.column(0).clone(),
                    options: Some(options),
                },
                SortColumn {
                    values: page.column(1).clone(),
                    options: None,
                },
            ],
            None,
        )?;
        let page = take_record_batch(&page, &sorted)?;
        Ok(RecordBatch::try_new(schema, page.columns().to_vec())?)
    }
}

impl RowSelection {
    fn select(&self, page: RecordBatch) -> Result<RecordBatch> {
        let keys = page.column(0);
        let mask = match self {
            Self::All => return Ok(page),
            Self::Nulls => is_null(keys)?,
            Self::Values(None) => is_not_null(keys)?,
            Self::Values(Some(predicate)) => {
                let matched = predicate.evaluate(&page)?.into_array(page.num_rows())?;
                and(&is_not_null(keys)?, matched.as_boolean())?
            }
        };
        Ok(filter_record_batch(&page, &mask)?)
    }

    /// The selection that reads one page for both runs, if one covers both.
    fn union(&self, other: &Self) -> Option<Self> {
        match (self, other) {
            (Self::Nulls, Self::Values(None)) | (Self::Values(None), Self::Nulls) => {
                Some(Self::All)
            }
            _ => None,
        }
    }
}

fn is_null_bound(bound: &Bound<ScalarValue>) -> bool {
    matches!(bound, Bound::Included(value) | Bound::Excluded(value) if value.is_null())
}

/// Page bounds and the page predicate compare only arrays of one type, so a
/// string literal of another string type is cast to the type of a string key.
fn strings_in_key_type(query: &SargableQuery, key_type: &DataType) -> Result<SargableQuery> {
    let is_string = |data_type: &DataType| {
        matches!(
            data_type,
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
        )
    };
    let cast = |value: &ScalarValue| -> Result<ScalarValue> {
        if is_string(key_type) && is_string(&value.data_type()) {
            Ok(value.cast_to(key_type)?)
        } else {
            Ok(value.clone())
        }
    };
    let cast_bound = |bound: &Bound<ScalarValue>| match bound {
        Bound::Included(value) => cast(value).map(Bound::Included),
        Bound::Excluded(value) => cast(value).map(Bound::Excluded),
        Bound::Unbounded => Ok(Bound::Unbounded),
    };
    Ok(match query {
        SargableQuery::Equals(value) => SargableQuery::Equals(cast(value)?),
        SargableQuery::Range(start, end) => {
            SargableQuery::Range(cast_bound(start)?, cast_bound(end)?)
        }
        SargableQuery::IsIn(values) => {
            SargableQuery::IsIn(values.iter().map(cast).collect::<Result<_>>()?)
        }
        SargableQuery::LikePrefix(prefix) => SargableQuery::LikePrefix(cast(prefix)?),
        SargableQuery::IsNull() | SargableQuery::FullTextSearch(_) => query.clone(),
    })
}

fn null_pages(lookup: CheckedLookup<'_>) -> Visits {
    let (all_null, partly_null) = (lookup.0.clone(), lookup.0.clone());
    Box::new(
        (0..all_null.all_null_pages.len())
            .map(move |i| PageVisit {
                page: all_null.all_null_pages[i],
                rows: RowSelection::All,
            })
            .chain((0..partly_null.null_pages.len()).map(move |i| PageVisit {
                page: partly_null.null_pages[i],
                rows: RowSelection::Nulls,
            })),
    )
}

impl ValuePages {
    fn between(
        lookup: CheckedLookup<'_>,
        lower: Bound<&ScalarValue>,
        upper: Bound<&ScalarValue>,
    ) -> Result<Self> {
        let lookup = lookup.0;
        let (mins, maxs) = (lookup.batch.column(0), lookup.batch.column(1));
        let (start, end) = (lookup.search_start, lookup.batch.num_rows());
        if start >= end {
            return Ok(Self::Matched(Vec::new()));
        }
        let first_row = |column: &ArrayRef, value: &ScalarValue, holds: fn(Ordering) -> bool| {
            let value = value.to_array_of_size(1)?;
            let compare = make_comparator(column, &value, STORED_ORDER)?;
            Ok::<_, Error>(partition_point(start, end, |row| !holds(compare(row, 0))))
        };
        let (rows_start, covered_start) = match lower {
            Bound::Unbounded => (start, partition_point(start, end, |row| mins.is_null(row))),
            Bound::Included(value) => (
                first_row(maxs, value, Ordering::is_ge)?,
                first_row(mins, value, Ordering::is_ge)?,
            ),
            Bound::Excluded(value) => (
                first_row(maxs, value, Ordering::is_gt)?,
                first_row(mins, value, Ordering::is_gt)?,
            ),
        };
        let (rows_end, covered_end) = match upper {
            Bound::Unbounded => (end, end),
            Bound::Included(value) => (
                first_row(mins, value, Ordering::is_gt)?,
                first_row(maxs, value, Ordering::is_gt)?,
            ),
            Bound::Excluded(value) => (
                first_row(mins, value, Ordering::is_ge)?,
                first_row(maxs, value, Ordering::is_ge)?,
            ),
        };
        Ok(Self::Rows {
            rows: rows_start..rows_end,
            covered: covered_start..covered_end,
        })
    }

    fn is_empty(&self) -> bool {
        match self {
            Self::Rows { rows, .. } => rows.is_empty(),
            Self::Matched(pages) => pages.is_empty(),
        }
    }

    /// Page id order is lookup order once [`check_page_order`] passes.
    fn visits(
        self,
        lookup: CheckedLookup<'_>,
        full: RowSelection,
        partial: RowSelection,
    ) -> Result<Visits> {
        let select = move |is_full: bool| {
            if is_full {
                full.clone()
            } else {
                partial.clone()
            }
        };
        Ok(match self {
            Self::Rows { rows, covered } => {
                let page_ids = lookup.0.page_numbers()?.clone();
                Box::new(rows.map(move |row| PageVisit {
                    page: page_ids.value(row),
                    rows: select(covered.contains(&row)),
                }))
            }
            Self::Matched(mut pages) => {
                pages.sort_unstable_by_key(Matches::page_id);
                Box::new(pages.into_iter().map(move |matches| PageVisit {
                    page: matches.page_id(),
                    rows: select(matches!(matches, Matches::All(_))),
                }))
            }
        })
    }
}

#[derive(Debug, Default)]
pub(super) struct PageOrderVerdict(OnceLock<Option<String>>);

impl DeepSizeOf for PageOrderVerdict {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.0
            .get()
            .map_or(0, |violation| violation.deep_size_of_children(context))
    }
}

fn check_page_order<'a>(
    lookup: &'a Arc<BTreeLookup>,
    metrics: &dyn MetricsCollector,
) -> Result<CheckedLookup<'a>> {
    let violation = match lookup.page_order.0.get() {
        Some(violation) => violation,
        None => {
            let (violation, rows_compared) = page_order_violation(lookup)?;
            metrics.record_comparisons(rows_compared);
            lookup.page_order.0.get_or_init(|| violation)
        }
    };
    match violation {
        Some(violation) => Err(Error::not_supported(format!(
            "ordered search needs B-tree pages in key order, but {violation}"
        ))),
        None => Ok(CheckedLookup(lookup)),
    }
}

/// A merge of range-partitioned segments concatenates lookups without
/// checking key order, and training on input sorted nulls last leaves a page
/// whose min sorts after its max.
fn page_order_violation(lookup: &BTreeLookup) -> Result<(Option<String>, usize)> {
    let (mins, maxs) = (lookup.batch.column(0), lookup.batch.column(1));
    let page_ids = lookup.page_numbers()?.values();
    let min_vs_max = make_comparator(mins, maxs, STORED_ORDER)?;
    let max_vs_next_min = make_comparator(maxs, mins, STORED_ORDER)?;
    let value = |array: &ArrayRef, row: usize| ScalarValue::try_from_array(array, row);
    for row in 0..lookup.batch.num_rows() {
        let page = page_ids[row];
        let violation = if min_vs_max(row, row) == Ordering::Greater {
            format!(
                "min > max: page {page} at lookup row {row} has min {} and max {}",
                value(mins, row)?,
                value(maxs, row)?
            )
        } else if row > 0 && page_ids[row - 1] >= page {
            format!(
                "page ids not increasing: page {} at lookup row {} precedes page {page} at \
                 lookup row {row}",
                page_ids[row - 1],
                row - 1
            )
        } else if row > 0 && max_vs_next_min(row - 1, row) == Ordering::Greater {
            format!(
                "previous max above current min: page {} at lookup row {} has max {}, page \
                 {page} at lookup row {row} has min {}",
                page_ids[row - 1],
                row - 1,
                value(maxs, row - 1)?,
                value(mins, row)?
            )
        } else {
            continue;
        };
        return Ok((Some(violation), row + 1));
    }
    Ok((None, lookup.batch.num_rows()))
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;
    use std::collections::HashMap;
    use std::ops::Bound;
    use std::sync::Arc;
    use std::sync::atomic::Ordering as AtomicOrdering;

    use arrow::compute::cast;
    use arrow_array::types::{Int32Type, UInt64Type};
    use arrow_array::{
        ArrayRef, Int32Array, RecordBatch, StringArray, UInt32Array, UInt64Array, cast::AsArray,
    };
    use arrow_schema::{DataType, Field, Schema, SortOptions};
    use datafusion::physical_plan::{SendableRecordBatchStream, stream::RecordBatchStreamAdapter};
    use datafusion_common::ScalarValue;
    use futures::{TryStreamExt, stream};
    use lance_core::cache::LanceCache;
    use lance_core::utils::tempfile::TempObjDir;
    use lance_core::{Error, ROW_ADDR, ROW_ID};
    use lance_io::object_store::ObjectStore;
    use rstest::rstest;
    use uuid::Uuid;

    use super::super::{
        BTREE_PAGES_NAME, BTreeIndex, Matches, OrderableScalarValue, merge_metadata_files,
        part_lookup_file_path, part_page_data_file_path, train_btree_index,
    };
    use super::{READ_AHEAD_PAGES, RowSelection, ValuePages, check_page_order};
    use crate::frag_reuse::{FragReuseIndex, FragReuseIndexDetails, FragReuseIndexHandle};
    use crate::metrics::{LocalMetricsCollector, MetricsCollector, NoOpMetricsCollector};
    use crate::progress::noop_progress;
    use crate::scalar::lance_format::LanceIndexStore;
    use crate::scalar::registry::VALUE_COLUMN_NAME;
    use crate::scalar::{AnyQuery, FullTextSearchQuery, IndexStore, SargableQuery, ScalarIndex};

    type Row = (Option<i32>, u64);

    fn batch_stream(batch: RecordBatch) -> SendableRecordBatchStream {
        let schema = batch.schema();
        let batches = if batch.num_rows() == 0 {
            Vec::new()
        } else {
            vec![Ok(batch)]
        };
        Box::pin(RecordBatchStreamAdapter::new(schema, stream::iter(batches)))
    }

    fn training_stream(rows: &[Row]) -> SendableRecordBatchStream {
        let schema = Arc::new(Schema::new(vec![
            Field::new(VALUE_COLUMN_NAME, DataType::Int32, true),
            Field::new(ROW_ADDR, DataType::UInt64, false),
        ]));
        batch_stream(
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(Int32Array::from_iter(rows.iter().map(|row| row.0))),
                    Arc::new(UInt64Array::from_iter_values(rows.iter().map(|row| row.1))),
                ],
            )
            .unwrap(),
        )
    }

    fn new_store() -> (Arc<LanceIndexStore>, TempObjDir) {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));
        (store, tmpdir)
    }

    async fn train(rows: &[Row], page_size: u64) -> (Arc<LanceIndexStore>, TempObjDir) {
        let mut sorted = rows.to_vec();
        sorted.sort_by_key(|row| row.0);
        let (store, tmpdir) = new_store();
        train_btree_index(
            training_stream(&sorted),
            store.as_ref(),
            page_size,
            None,
            None,
        )
        .await
        .unwrap();
        (store, tmpdir)
    }

    async fn build_index(rows: &[Row], page_size: u64) -> (Arc<BTreeIndex>, TempObjDir) {
        let (store, tmpdir) = train(rows, page_size).await;
        let index = BTreeIndex::load(store, None, &LanceCache::no_cache(), true)
            .await
            .unwrap();
        (index, tmpdir)
    }

    fn with_lookup_column(index: &BTreeIndex, column: usize, values: ArrayRef) -> BTreeIndex {
        let mut columns = index.page_lookup.batch.columns().to_vec();
        columns[column] = values;
        let lookup = RecordBatch::try_new(index.page_lookup.batch.schema(), columns).unwrap();
        BTreeIndex::try_from_serialized(
            lookup,
            index.store.clone(),
            &LanceCache::no_cache(),
            index.batch_size,
            index.ranges_to_files.clone(),
            None,
            index.results_are_row_addresses,
        )
        .unwrap()
    }

    fn order(descending: bool, nulls_first: bool) -> SortOptions {
        SortOptions {
            descending,
            nulls_first,
        }
    }

    async fn ordered_rows_with_metrics(
        index: &BTreeIndex,
        query: Option<&SargableQuery>,
        options: SortOptions,
        metrics: Arc<dyn MetricsCollector>,
    ) -> Vec<Row> {
        let batches: Vec<RecordBatch> = index
            .as_ordered()
            .unwrap()
            .search_ordered(query.map(|query| query as &dyn AnyQuery), options, metrics)
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        batches
            .iter()
            .flat_map(|batch| {
                let keys = batch.column(0).as_primitive::<Int32Type>();
                let ids = batch.column(1).as_primitive::<UInt64Type>();
                keys.iter().zip(ids.values().iter().copied())
            })
            .collect()
    }

    async fn ordered_rows(
        index: &BTreeIndex,
        query: Option<&SargableQuery>,
        options: SortOptions,
    ) -> Vec<Row> {
        ordered_rows_with_metrics(index, query, options, Arc::new(NoOpMetricsCollector)).await
    }

    fn keys(rows: &[Row]) -> Vec<Option<i32>> {
        rows.iter().map(|row| row.0).collect()
    }

    fn parts_loaded(metrics: &LocalMetricsCollector) -> usize {
        metrics.parts_loaded.load(AtomicOrdering::Relaxed)
    }

    fn key_order(a: Option<i32>, b: Option<i32>, options: SortOptions) -> Ordering {
        match (a, b) {
            (None, None) => Ordering::Equal,
            (None, Some(_)) if options.nulls_first => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (Some(_), None) => key_order(b, a, options).reverse(),
            (Some(a), Some(b)) if options.descending => b.cmp(&a),
            (Some(a), Some(b)) => a.cmp(&b),
        }
    }

    fn sort_rows(rows: &mut [Row], options: SortOptions) {
        rows.sort_by(|a, b| key_order(a.0, b.0, options).then(a.1.cmp(&b.1)));
    }

    async fn sorted_search_rows(
        index: &BTreeIndex,
        rows: &[Row],
        query: Option<&SargableQuery>,
        options: SortOptions,
    ) -> Vec<Row> {
        let mut expected = match query {
            None => rows.to_vec(),
            Some(query) => {
                let key_of: HashMap<u64, Option<i32>> =
                    rows.iter().map(|&(key, id)| (id, key)).collect();
                let result = index.search(query, &NoOpMetricsCollector).await.unwrap();
                result
                    .row_addrs()
                    .true_rows()
                    .row_addrs()
                    .unwrap()
                    .map(|id| {
                        let id = u64::from(id);
                        (key_of[&id], id)
                    })
                    .collect()
            }
        };
        sort_rows(&mut expected, options);
        expected
    }

    /// Equal keys on different pages have no id order, so ids are compared
    /// once both sides are sorted.
    fn assert_same_rows(mut actual: Vec<Row>, expected: &[Row], options: SortOptions, case: &str) {
        assert_eq!(keys(&actual), keys(expected), "keys out of order, {case}");
        sort_rows(&mut actual, options);
        assert_eq!(
            actual, expected,
            "a row is missing, repeated, or under the wrong key, {case}"
        );
    }

    #[derive(Clone, Copy, Debug)]
    enum Fixture {
        Paged(u64),
        SinglePage,
        AllNull,
        Empty,
    }

    impl Fixture {
        /// Ties cross page boundaries, and a tie's ids do not ascend in
        /// training order.
        fn rows_and_page_size(self) -> (Vec<Row>, u64) {
            const TIE_RUN: u64 = 3;
            const NULL_EVERY: u64 = 10;
            let (num_rows, page_size, has_keys) = match self {
                Self::Paged(page_size) => (2 * page_size + 50, page_size, true),
                Self::SinglePage => (20, 4096, true),
                Self::AllNull => (30, 7, false),
                Self::Empty => (0, 7, true),
            };
            let rows = (0..num_rows)
                .map(|i| {
                    let key = (has_keys && i % NULL_EVERY != 3).then_some((i / TIE_RUN) as i32);
                    // An odd multiplier permutes u32, so ids stay unique.
                    let id = (i as u32).wrapping_mul(2_654_435_761);
                    (key, (u64::from(id % 4) << 32) | u64::from(id))
                })
                .collect();
            (rows, page_size)
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum QueryKind {
        AllRows,
        Equals,
        Range,
        Unbounded,
        In,
        IsNull,
    }

    impl QueryKind {
        fn query(self, max_key: i32) -> Option<SargableQuery> {
            let key = |key: i32| ScalarValue::Int32(Some(key));
            match self {
                Self::AllRows => None,
                Self::Equals => Some(SargableQuery::Equals(key(max_key / 2))),
                Self::Range => Some(SargableQuery::Range(
                    Bound::Included(key(max_key / 4)),
                    Bound::Excluded(key(3 * max_key / 4)),
                )),
                Self::Unbounded => Some(SargableQuery::Range(Bound::Unbounded, Bound::Unbounded)),
                Self::In => Some(SargableQuery::IsIn(vec![
                    key(1),
                    key(max_key / 2),
                    key(max_key / 2),
                    key(max_key + 5),
                    ScalarValue::Int32(None),
                    key(max_key),
                ])),
                Self::IsNull => Some(SargableQuery::IsNull()),
            }
        }
    }

    #[rstest]
    #[case::asc_nulls_first(false, true)]
    #[case::asc_nulls_last(false, false)]
    #[case::desc_nulls_first(true, true)]
    #[case::desc_nulls_last(true, false)]
    #[tokio::test]
    async fn test_ordered_matches_sorted_search(
        #[case] descending: bool,
        #[case] nulls_first: bool,
        #[values(
            Fixture::Paged(1),
            Fixture::Paged(7),
            Fixture::Paged(4096),
            Fixture::SinglePage,
            Fixture::AllNull,
            Fixture::Empty
        )]
        fixture: Fixture,
        #[values(
            QueryKind::AllRows,
            QueryKind::Equals,
            QueryKind::Range,
            QueryKind::Unbounded,
            QueryKind::In,
            QueryKind::IsNull
        )]
        query: QueryKind,
    ) {
        let (rows, page_size) = fixture.rows_and_page_size();
        let (index, _tmpdir) = build_index(&rows, page_size).await;
        let max_key = rows.iter().filter_map(|row| row.0).max().unwrap_or(0);
        let query = query.query(max_key);
        let options = order(descending, nulls_first);

        let actual = ordered_rows(&index, query.as_ref(), options).await;
        let expected = sorted_search_rows(&index, &rows, query.as_ref(), options).await;

        assert_same_rows(actual, &expected, options, "ordered search vs search");
    }

    #[rstest]
    #[case::ascending(
        false,
        true,
        None,
        vec![None, None, None, Some(1), Some(1), Some(1), Some(2)],
        vec![10, 20, 30, 40, 50, 60, 5]
    )]
    #[case::descending(
        true,
        false,
        None,
        vec![Some(2), Some(1), Some(1), Some(1), None, None, None],
        vec![5, 40, 50, 60, 10, 20, 30]
    )]
    #[case::descending_range(
        true,
        false,
        Some(SargableQuery::Range(
            Bound::Included(ScalarValue::Int32(Some(1))),
            Bound::Included(ScalarValue::Int32(Some(2))),
        )),
        vec![Some(2), Some(1), Some(1), Some(1)],
        vec![5, 40, 50, 60]
    )]
    #[tokio::test]
    async fn test_ordered_ties_are_row_address_ascending(
        #[case] descending: bool,
        #[case] nulls_first: bool,
        #[case] query: Option<SargableQuery>,
        #[case] expected_keys: Vec<Option<i32>>,
        #[case] expected_ids: Vec<u64>,
    ) {
        let rows = [
            (None, 30),
            (None, 10),
            (None, 20),
            (Some(1), 60),
            (Some(1), 40),
            (Some(1), 50),
            (Some(2), 5),
        ];
        let (index, _tmpdir) = build_index(&rows, 16).await;

        let actual = ordered_rows(&index, query.as_ref(), order(descending, nulls_first)).await;

        assert_eq!(keys(&actual), expected_keys);
        assert_eq!(
            actual.iter().map(|row| row.1).collect::<Vec<_>>(),
            expected_ids,
            "equal keys come in ascending row address order in both directions"
        );
    }

    #[rstest]
    #[case::asc_nulls_first(false, true)]
    #[case::asc_nulls_last(false, false)]
    #[case::desc_nulls_first(true, true)]
    #[case::desc_nulls_last(true, false)]
    #[tokio::test]
    async fn test_ordered_all_rows_null_complete_full_and_paged(
        #[case] descending: bool,
        #[case] nulls_first: bool,
    ) {
        let rows: Vec<Row> = [None, None, None, None]
            .into_iter()
            .chain([Some(1), Some(1), Some(1), Some(1), Some(2), Some(3)])
            .zip(100..)
            .collect();
        let range = SargableQuery::Range(
            Bound::Included(ScalarValue::Int32(Some(1))),
            Bound::Included(ScalarValue::Int32(Some(2))),
        );
        let unbounded = SargableQuery::Range(Bound::Unbounded, Bound::Unbounded);
        let options = order(descending, nulls_first);

        const ONE_PAGE: u64 = 16;
        const STRADDLING_PAGES: u64 = 3;
        let (one_page, _one_page_dir) = build_index(&rows, ONE_PAGE).await;
        let (paged, _paged_dir) = build_index(&rows, STRADDLING_PAGES).await;
        // Before Lance 0.25.2, the min of the straddling page [N 1 1] skipped
        // its null and recorded 1.
        let legacy_mins: Int32Array = paged
            .page_lookup
            .batch
            .column(0)
            .as_primitive::<Int32Type>()
            .iter()
            .zip(
                paged
                    .page_lookup
                    .batch
                    .column(1)
                    .as_primitive::<Int32Type>(),
            )
            .map(|(min, max)| match (min, max) {
                (None, Some(_)) => Some(1),
                _ => min,
            })
            .collect();
        let legacy = with_lookup_column(&paged, 0, Arc::new(legacy_mins));
        // On legacy page stats, search reports the straddling page's null as
        // NULL, so the oracle is search over the current stats of the same rows.
        let all_rows = sorted_search_rows(&paged, &rows, Some(&unbounded), options).await;
        let in_range = sorted_search_rows(&paged, &rows, Some(&range), options).await;
        assert_eq!(
            all_rows.len(),
            rows.len(),
            "an unbounded range selects every row"
        );
        // Four pages, and the straddling page is read twice when the walk
        // separates its nulls from its keys.
        let paged_reads = 4 + usize::from(nulls_first == descending);

        for (name, index, expected_reads) in [
            ("one page", one_page.as_ref(), 1),
            ("straddling pages", paged.as_ref(), paged_reads),
            ("legacy page stats", &legacy, paged_reads),
        ] {
            let metrics = Arc::new(LocalMetricsCollector::default());
            let actual = ordered_rows_with_metrics(index, None, options, metrics.clone()).await;
            assert_same_rows(actual, &all_rows, options, &format!("all rows, {name}"));
            assert_eq!(parts_loaded(&metrics), expected_reads, "pages read, {name}");
            assert_same_rows(
                ordered_rows(index, Some(&unbounded), options).await,
                &all_rows,
                options,
                &format!("unbounded range, {name}"),
            );
            assert_same_rows(
                ordered_rows(index, Some(&range), options).await,
                &in_range,
                options,
                &format!("bounded range, {name}"),
            );
        }
    }

    #[tokio::test]
    async fn test_ordered_range_walk_picks_the_pages_search_picks() {
        let mut indices = Vec::new();
        let mut dirs = Vec::new();
        for fixture in [
            Fixture::Paged(1),
            Fixture::Paged(7),
            Fixture::Paged(4096),
            Fixture::SinglePage,
            Fixture::AllNull,
            Fixture::Empty,
        ] {
            let (rows, page_size) = fixture.rows_and_page_size();
            let (index, dir) = build_index(&rows, page_size).await;
            indices.push((format!("{fixture:?}"), index));
            dirs.push(dir);
        }
        let ties: Vec<Row> = (0..240).map(|i| (Some(i as i32 / 60), i)).collect();
        let (index, dir) = build_index(&ties, 4).await;
        indices.push(("ties across pages".to_string(), index));
        dirs.push(dir);
        let straddling: Vec<Row> = [None, None, None, None]
            .into_iter()
            .chain([Some(1), Some(1), Some(1), Some(1), Some(2), Some(3)])
            .zip(100..)
            .collect();
        let (paged, dir) = build_index(&straddling, 3).await;
        dirs.push(dir);
        assert_eq!(
            paged
                .page_lookup
                .batch
                .column(0)
                .as_primitive::<Int32Type>(),
            &Int32Array::from(vec![None, None, Some(1), Some(3)]),
            "pages [N N N] [N 1 1] [1 1 2] [3]"
        );
        let legacy_mins = Int32Array::from(vec![None, Some(1), Some(1), Some(3)]);
        let legacy = with_lookup_column(&paged, 0, Arc::new(legacy_mins));
        indices.push(("legacy page stats".to_string(), Arc::new(legacy)));

        let mut pages_at_excluded_upper = 0;
        for (name, index) in &indices {
            let lookup = check_page_order(&index.page_lookup, &NoOpMetricsCollector).unwrap();
            let batch = &index.page_lookup.batch;
            let page_min: HashMap<u32, Option<i32>> = index
                .page_lookup
                .page_numbers()
                .unwrap()
                .values()
                .iter()
                .copied()
                .zip(batch.column(0).as_primitive::<Int32Type>().iter())
                .collect();
            let mut keys: Vec<i32> = [batch.column(0), batch.column(1)]
                .into_iter()
                .flat_map(|column| column.as_primitive::<Int32Type>().iter().flatten())
                .flat_map(|key| [key - 1, key, key + 1])
                .collect();
            keys.sort_unstable();
            keys.dedup();
            let bounds: Vec<Bound<ScalarValue>> = std::iter::once(Bound::Unbounded)
                .chain(keys.iter().flat_map(|&key| {
                    let key = ScalarValue::Int32(Some(key));
                    [Bound::Included(key.clone()), Bound::Excluded(key)]
                }))
                .chain([Bound::Included(ScalarValue::Int64(Some(0)))])
                .collect();
            for lower in &bounds {
                for upper in &bounds {
                    let key = |bound: &Bound<ScalarValue>| bound.clone().map(OrderableScalarValue);
                    let expected = index
                        .page_lookup
                        .pages_between((key(lower).as_ref(), key(upper).as_ref()));
                    let walked = ValuePages::between(lookup, lower.as_ref(), upper.as_ref())
                        .and_then(|pages| {
                            pages.visits(lookup, RowSelection::All, RowSelection::Nulls)
                        })
                        .map(|visits| {
                            visits
                                .map(|visit| match visit.rows {
                                    RowSelection::All => Matches::All(visit.page),
                                    _ => Matches::Some(visit.page),
                                })
                                .collect::<Vec<_>>()
                        });
                    match (walked, expected) {
                        (Ok(walked), Ok(expected)) => {
                            let (at_excluded_upper, expected_walk): (Vec<_>, Vec<_>) =
                                expected.into_iter().partition(|page| {
                                    matches!(upper, Bound::Excluded(ScalarValue::Int32(bound))
                                        if page_min[&page.page_id()] == *bound)
                                });
                            pages_at_excluded_upper += at_excluded_upper.len();
                            assert_eq!(
                                walked, expected_walk,
                                "{name}, {lower:?} to {upper:?}: the walk skips only the pages \
                                 search keeps whose min is the excluded upper bound"
                            )
                        }
                        (Err(_), Err(_)) => {}
                        (walked, expected) => panic!(
                            "{name}, {lower:?} to {upper:?}: walk {walked:?}, search {expected:?}"
                        ),
                    }
                }
            }
        }
        assert!(
            pages_at_excluded_upper > 0,
            "some range has a page whose min is its excluded upper bound"
        );
    }

    #[rstest]
    #[case::ascending(false)]
    #[case::descending(true)]
    #[tokio::test]
    async fn test_ordered_exclusive_upper_skips_pages_at_the_bound(#[case] descending: bool) {
        const ROWS_PER_KEY: u64 = 41;
        const BOUND: i32 = 2;
        let rows: Vec<Row> = (0..4 * ROWS_PER_KEY)
            .map(|i| (Some((i / ROWS_PER_KEY) as i32), i))
            .collect();
        let (index, _tmpdir) = build_index(&rows, 4).await;
        let mins = index
            .page_lookup
            .batch
            .column(0)
            .as_primitive::<Int32Type>();
        let pages_below_bound = mins.iter().filter(|&min| min < Some(BOUND)).count();
        assert_eq!(
            mins.iter().filter(|&min| min == Some(BOUND)).count(),
            10,
            "min 2 on nine [2 2 2 2] pages and on [2 2 2 3]"
        );
        let query = SargableQuery::Range(
            Bound::Unbounded,
            Bound::Excluded(ScalarValue::Int32(Some(BOUND))),
        );
        let options = order(descending, true);

        let first_row = Arc::new(LocalMetricsCollector::default());
        let mut stream = index
            .as_ordered()
            .unwrap()
            .search_ordered(Some(&query), options, first_row.clone())
            .unwrap();
        stream.try_next().await.unwrap().unwrap();
        assert_eq!(
            parts_loaded(&first_row),
            1,
            "pages read before the first row"
        );

        let full_walk = Arc::new(LocalMetricsCollector::default());
        let actual =
            ordered_rows_with_metrics(&index, Some(&query), options, full_walk.clone()).await;
        let expected = sorted_search_rows(&index, &rows, Some(&query), options).await;
        assert_same_rows(actual, &expected, options, "x < bound");
        assert_eq!(
            parts_loaded(&full_walk),
            pages_below_bound,
            "a full walk reads only the pages whose min is below the bound"
        );
    }

    #[rstest]
    #[case::ascending(false, vec![Some(0), Some(20), Some(39)])]
    #[case::descending(true, vec![Some(39), Some(20), Some(0)])]
    #[tokio::test]
    async fn test_ordered_in_list_skips_null_only_pages(
        #[case] descending: bool,
        #[case] expected: Vec<Option<i32>>,
    ) {
        const PAGE_SIZE: u64 = 4;
        let null_rows = 2 * PAGE_SIZE;
        let rows: Vec<Row> = (0..null_rows + 40)
            .map(|i| ((i >= null_rows).then(|| (i - null_rows) as i32), i))
            .collect();
        let (index, _tmpdir) = build_index(&rows, PAGE_SIZE).await;
        let key = |key: Option<i32>| ScalarValue::Int32(key);
        let query = SargableQuery::IsIn(vec![
            key(Some(39)),
            key(Some(0)),
            key(Some(20)),
            key(Some(20)),
            key(None),
        ]);
        let metrics = Arc::new(LocalMetricsCollector::default());

        let actual = ordered_rows_with_metrics(
            &index,
            Some(&query),
            order(descending, true),
            metrics.clone(),
        )
        .await;

        assert_eq!(keys(&actual), expected);
        assert_eq!(
            parts_loaded(&metrics),
            3,
            "only the pages holding 0, 20, and 39 are read"
        );
    }

    #[rstest]
    #[case::equals_null(SargableQuery::Equals(ScalarValue::Int32(None)))]
    #[case::null_lower_bound(SargableQuery::Range(
        Bound::Included(ScalarValue::Int32(None)),
        Bound::Unbounded
    ))]
    #[case::null_upper_bound(SargableQuery::Range(
        Bound::Unbounded,
        Bound::Excluded(ScalarValue::Int32(None))
    ))]
    #[case::in_only_null(SargableQuery::IsIn(vec![ScalarValue::Int32(None)]))]
    #[case::null_string_prefix(SargableQuery::LikePrefix(ScalarValue::Utf8(None)))]
    #[tokio::test]
    async fn test_ordered_null_operands_read_nothing(#[case] query: SargableQuery) {
        let rows: Vec<Row> = (0..40u64)
            .map(|i| ((i % 4 != 0).then_some(i as i32), i))
            .collect();
        let (index, _tmpdir) = build_index(&rows, 4).await;
        let metrics = Arc::new(LocalMetricsCollector::default());

        let actual =
            ordered_rows_with_metrics(&index, Some(&query), order(false, true), metrics.clone())
                .await;

        assert_eq!(actual, Vec::<Row>::new());
        assert_eq!(parts_loaded(&metrics), 0);
    }

    #[derive(Clone, Copy, Debug)]
    enum StringQuery {
        LikePrefix,
        Equals,
        Range,
        In,
    }

    impl StringQuery {
        fn query(self, literal_type: &DataType) -> SargableQuery {
            let literal = |text: &str| ScalarValue::from(text).cast_to(literal_type).unwrap();
            match self {
                Self::LikePrefix => SargableQuery::LikePrefix(literal("ap")),
                Self::Equals => SargableQuery::Equals(literal("apple")),
                Self::Range => SargableQuery::Range(
                    Bound::Included(literal("ap")),
                    Bound::Excluded(literal("aq")),
                ),
                Self::In => {
                    SargableQuery::IsIn(vec![literal("apple"), literal("zebra"), literal("ap")])
                }
            }
        }
    }

    /// B-tree training rejects `Utf8View` keys. Every query matches "ap",
    /// whose page also holds keys the query rejects, so the page predicate runs.
    #[rstest]
    #[tokio::test]
    async fn test_ordered_string_literal_types(
        #[values(false, true)] descending: bool,
        #[values(DataType::Utf8, DataType::LargeUtf8)] key_type: DataType,
        #[values(DataType::Utf8, DataType::LargeUtf8, DataType::Utf8View)] literal_type: DataType,
        #[values(
            StringQuery::LikePrefix,
            StringQuery::Equals,
            StringQuery::Range,
            StringQuery::In
        )]
        query: StringQuery,
    ) {
        let keys = [
            None,
            Some("aardvark"),
            Some("ap"),
            Some("apex"),
            Some("apple"),
            Some("apricot"),
            Some("aq"),
            Some("banana"),
        ];
        let schema = Arc::new(Schema::new(vec![
            Field::new(VALUE_COLUMN_NAME, key_type.clone(), true),
            Field::new(ROW_ADDR, DataType::UInt64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                cast(&StringArray::from(keys.to_vec()), &key_type).unwrap(),
                Arc::new(UInt64Array::from_iter_values(0..keys.len() as u64)),
            ],
        )
        .unwrap();
        let (store, _tmpdir) = new_store();
        train_btree_index(batch_stream(batch), store.as_ref(), 3, None, None)
            .await
            .unwrap();
        let index = BTreeIndex::load(store, None, &LanceCache::no_cache(), true)
            .await
            .unwrap();
        let options = order(descending, true);

        let batches: Vec<RecordBatch> = index
            .as_ordered()
            .unwrap()
            .search_ordered(
                Some(&query.query(&literal_type)),
                options,
                Arc::new(NoOpMetricsCollector),
            )
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let actual: Vec<u64> = batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column(1)
                    .as_primitive::<UInt64Type>()
                    .values()
                    .to_vec()
            })
            .collect();
        let mut expected: Vec<u64> = index
            .search(&query.query(&key_type), &NoOpMetricsCollector)
            .await
            .unwrap()
            .row_addrs()
            .true_rows()
            .row_addrs()
            .unwrap()
            .map(u64::from)
            .collect();
        expected.sort_by_key(|&id| keys[id as usize]);
        if descending {
            expected.reverse();
        }
        assert_eq!(
            actual, expected,
            "{query:?} with a {literal_type} literal on a {key_type} index"
        );
    }

    #[rstest]
    #[case::binary(ScalarValue::Binary(Some(b"ap".to_vec())))]
    #[case::null_binary(ScalarValue::Binary(None))]
    #[case::untyped_null(ScalarValue::Null)]
    #[tokio::test]
    async fn test_ordered_rejects_non_string_like_prefix(#[case] prefix: ScalarValue) {
        let (index, _tmpdir) = build_index(&[(Some(1), 0)], 16).await;
        let query = SargableQuery::LikePrefix(prefix.clone());

        let error = index
            .as_ordered()
            .unwrap()
            .search_ordered(
                Some(&query),
                order(false, true),
                Arc::new(NoOpMetricsCollector),
            )
            .err()
            .expect("a LIKE prefix that is not a string has no key range");

        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        let message = error.to_string();
        assert!(message.contains(&prefix.to_string()), "{message}");
        assert!(
            message.contains(&prefix.data_type().to_string()),
            "{message}"
        );
    }

    #[tokio::test]
    async fn test_ordered_rejects_full_text_search() {
        let (index, _tmpdir) = build_index(&[(Some(1), 0)], 16).await;
        let query = SargableQuery::FullTextSearch(FullTextSearchQuery::new("ap".to_string()));

        let error = index
            .as_ordered()
            .unwrap()
            .search_ordered(
                Some(&query),
                order(false, true),
                Arc::new(NoOpMetricsCollector),
            )
            .err()
            .expect("a B-tree cannot answer full text search");

        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains("an inverted index"), "{error}");
    }

    #[tokio::test]
    async fn test_ordered_applies_frag_reuse_remap() {
        let rows: Vec<Row> = (0..8).map(|i| (Some(i as i32), i)).collect();
        let (index, _tmpdir) = build_index(&rows, 4).await;
        let mut remapped = index.as_ref().clone();
        remapped.frag_reuse_index = Some(Arc::new(FragReuseIndexHandle(Arc::new(
            FragReuseIndex::new(
                Uuid::new_v4(),
                vec![HashMap::from([(0, Some(5000)), (1, None)])],
                FragReuseIndexDetails { versions: vec![] },
            ),
        ))));

        let actual = ordered_rows(&remapped, None, order(false, true)).await;

        let expected: Vec<Row> = [(Some(0), 5000)]
            .into_iter()
            .chain((2..8).map(|i| (Some(i as i32), i)))
            .collect();
        assert_eq!(actual, expected);
    }

    #[rstest]
    #[case::in_key_order(0, 100, true, &[])]
    #[case::out_of_key_order(100, 0, false, &[
        "previous max above current min",
        "page 6 at lookup row 6",
        "max 199",
        "page 7 at lookup row 7",
        "min 0",
    ])]
    #[tokio::test]
    async fn test_ordered_range_partitioned_lookup(
        #[case] first_partition_key: i32,
        #[case] second_partition_key: i32,
        #[case] supported: bool,
        #[case] violation: &[&str],
    ) {
        let (store, _tmpdir) = new_store();
        for (partition, first_key) in [(0u32, first_partition_key), (1, second_partition_key)] {
            let rows: Vec<Row> = (first_key..first_key + 100)
                .map(|key| (Some(key), key as u64))
                .collect();
            train_btree_index(
                training_stream(&rows),
                store.as_ref(),
                16,
                None,
                Some(partition),
            )
            .await
            .unwrap();
        }
        merge_metadata_files(
            store.as_ref(),
            &[
                part_page_data_file_path(0),
                part_page_data_file_path(1 << 32),
            ],
            &[part_lookup_file_path(0), part_lookup_file_path(1 << 32)],
            Some(1),
            noop_progress(),
        )
        .await
        .unwrap();
        let index = BTreeIndex::load(store, None, &LanceCache::no_cache(), true)
            .await
            .unwrap();
        let options = order(false, true);

        if supported {
            assert_eq!(
                keys(&ordered_rows(&index, None, options).await),
                (0..200).map(Some).collect::<Vec<_>>()
            );
        } else {
            let error = index
                .as_ordered()
                .unwrap()
                .search_ordered(None, options, Arc::new(NoOpMetricsCollector))
                .err()
                .expect("an out-of-order lookup must not yield a stream");
            assert!(matches!(error, Error::NotSupported { .. }), "{error}");
            for fragment in violation {
                assert!(error.to_string().contains(fragment), "{fragment}: {error}");
            }
        }
    }

    #[tokio::test]
    async fn test_ordered_rejects_pages_out_of_key_order() {
        let rows: Vec<Row> = (0..10).map(Some).chain([None, None]).zip(0..).collect();
        let (store, _tmpdir) = new_store();
        train_btree_index(training_stream(&rows), store.as_ref(), 4, None, None)
            .await
            .unwrap();
        let trained_nulls_last = BTreeIndex::load(store, None, &LanceCache::no_cache(), true)
            .await
            .unwrap();
        let (in_order, _in_order_dir) = build_index(&rows, 4).await;
        let swapped_ids: Vec<u32> = [1, 0]
            .into_iter()
            .chain(
                in_order.page_lookup.page_numbers().unwrap().values()[2..]
                    .iter()
                    .copied(),
            )
            .collect();
        let swapped_pages =
            with_lookup_column(&in_order, 3, Arc::new(UInt32Array::from(swapped_ids)));

        // Training records a page's first and last keys, so the nulls-last page
        // [8 9 N N] has min 8 and max NULL.
        for (name, index, violation) in [
            (
                "trained nulls last",
                trained_nulls_last.as_ref(),
                ["min > max", "page 2 at lookup row 2", "min 8", "max NULL"].as_slice(),
            ),
            (
                "page ids out of order",
                &swapped_pages,
                &[
                    "page ids not increasing",
                    "page 1 at lookup row 0",
                    "page 0 at lookup row 1",
                ],
            ),
        ] {
            let error = index
                .as_ordered()
                .unwrap()
                .search_ordered(None, order(false, true), Arc::new(NoOpMetricsCollector))
                .err()
                .unwrap_or_else(|| {
                    panic!("{name}: an out-of-order lookup must not yield a stream")
                });
            assert!(
                matches!(error, Error::NotSupported { .. }),
                "{name}: {error}"
            );
            for fragment in violation {
                assert!(
                    error.to_string().contains(fragment),
                    "{name}, {fragment}: {error}"
                );
            }
        }
    }

    #[rstest]
    #[case::in_key_order(false, 10)]
    #[case::out_of_key_order(true, 2)]
    #[tokio::test]
    async fn test_ordered_checks_page_order_once_per_lookup(
        #[case] pages_swapped: bool,
        #[case] rows_compared: usize,
    ) {
        let rows: Vec<Row> = (0..40).map(|i| (Some(i), i as u64)).collect();
        let (in_order, _tmpdir) = build_index(&rows, 4).await;
        let page_ids = in_order
            .page_lookup
            .page_numbers()
            .unwrap()
            .values()
            .to_vec();
        let index = if pages_swapped {
            let swapped: Vec<u32> = [page_ids[1], page_ids[0]]
                .into_iter()
                .chain(page_ids[2..].iter().copied())
                .collect();
            with_lookup_column(&in_order, 3, Arc::new(UInt32Array::from(swapped)))
        } else {
            in_order.as_ref().clone()
        };
        index
            .search(
                &SargableQuery::Equals(ScalarValue::Int32(Some(5))),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();

        let mut comparisons = Vec::new();
        let mut errors = Vec::new();
        for _ in 0..2 {
            let metrics = Arc::new(LocalMetricsCollector::default());
            let result = index.as_ordered().unwrap().search_ordered(
                None,
                order(false, true),
                metrics.clone(),
            );
            comparisons.push(metrics.comparisons.load(AtomicOrdering::Relaxed));
            errors.push(result.err().map(|error| {
                assert!(matches!(error, Error::NotSupported { .. }), "{error}");
                error.to_string()
            }));
        }

        assert_eq!(
            comparisons,
            [rows_compared, 0],
            "a plain search leaves the lookup unchecked, and only the first ordered search checks it"
        );
        assert_eq!(errors[0].is_some(), pages_swapped);
        assert_eq!(
            errors[1], errors[0],
            "every ordered search gets the same verdict"
        );
    }

    #[tokio::test]
    async fn test_ordered_reads_pages_lazily() {
        let rows: Vec<Row> = (0..1600u64).map(|i| (Some(i as i32), i)).collect();
        let (index, _tmpdir) = build_index(&rows, 16).await;
        let options = order(false, true);

        let unpolled = Arc::new(LocalMetricsCollector::default());
        drop(
            index
                .as_ordered()
                .unwrap()
                .search_ordered(None, options, unpolled.clone())
                .unwrap(),
        );
        assert_eq!(parts_loaded(&unpolled), 0, "opening a stream reads no page");

        let first_ten = Arc::new(LocalMetricsCollector::default());
        let mut stream = index
            .as_ordered()
            .unwrap()
            .search_ordered(None, options, first_ten.clone())
            .unwrap();
        let mut taken = 0;
        while taken < 10 {
            taken += stream.try_next().await.unwrap().unwrap().num_rows();
        }
        drop(stream);
        let loaded = parts_loaded(&first_ten);
        assert!(
            (1..=3).contains(&loaded),
            "taking 10 rows from 100 pages loaded {loaded} pages"
        );

        let full = Arc::new(LocalMetricsCollector::default());
        let all = ordered_rows_with_metrics(&index, None, options, full.clone()).await;
        assert_eq!(all.len(), 1600);
        assert_eq!(parts_loaded(&full), 100, "a full walk reads each page once");
    }

    #[tokio::test]
    async fn test_ordered_read_ahead_starts_after_first_page() {
        // Smaller pages fit in the bytes the open already reads, so reading
        // them never reaches the object store.
        const PAGE_SIZE: u64 = 4096;
        let rows: Vec<Row> = (0..8 * PAGE_SIZE).map(|i| (Some(i as i32), i)).collect();
        let object_store = Arc::new(ObjectStore::local());
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            object_store.clone(),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));
        train_btree_index(
            training_stream(&rows),
            store.as_ref(),
            PAGE_SIZE,
            None,
            None,
        )
        .await
        .unwrap();
        let index = BTreeIndex::load(store.clone(), None, &LanceCache::no_cache(), true)
            .await
            .unwrap();
        let take_reads = || object_store.io_stats_incremental().read_iops;
        take_reads();
        let page_file = store.open_index_file(BTREE_PAGES_NAME).await.unwrap();
        let open_reads = take_reads();
        page_file.read_record_batch(0, PAGE_SIZE).await.unwrap();
        let page_reads = take_reads();
        assert!(page_reads > 0, "a page read reaches the object store");

        let mut stream = index
            .as_ordered()
            .unwrap()
            .search_ordered(None, order(false, true), Arc::new(NoOpMetricsCollector))
            .unwrap();
        stream.try_next().await.unwrap().unwrap();
        assert_eq!(
            take_reads(),
            open_reads + page_reads,
            "the first batch reads one page"
        );
        stream.try_next().await.unwrap().unwrap();
        let second = take_reads();
        assert!(
            (page_reads..=(1 + READ_AHEAD_PAGES as u64) * page_reads).contains(&second),
            "the second batch read {second} times, page reads cost {page_reads}"
        );
    }

    #[rstest]
    #[case::row_addresses(true, ROW_ADDR)]
    #[case::row_ids(false, ROW_ID)]
    #[tokio::test]
    async fn test_ordered_id_column_follows_id_domain(
        #[case] results_are_row_addresses: bool,
        #[case] expected: &str,
    ) {
        let (store, _tmpdir) = train(&[(Some(1), 7)], 16).await;
        let index = BTreeIndex::load(
            store,
            None,
            &LanceCache::no_cache(),
            results_are_row_addresses,
        )
        .await
        .unwrap();

        let stream = index
            .as_ordered()
            .unwrap()
            .search_ordered(None, order(false, true), Arc::new(NoOpMetricsCollector))
            .unwrap();

        assert_eq!(stream.schema().field(1).name(), expected);
    }
}
