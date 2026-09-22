// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_core::utils::row_addr_remap::RowAddrRemap;
use std::collections::BTreeSet;
use std::{ops::Bound, sync::Arc};

use arrow_array::Array;
use arrow_array::{
    ArrayRef, BooleanArray, RecordBatch, UInt64Array, cast::AsArray, types::UInt64Type,
};

use datafusion_common::DFSchema;
use datafusion_expr::execution_props::ExecutionProps;
use datafusion_physical_expr::{PhysicalExpr, create_physical_expr};
use lance_arrow::RecordBatchExt;
use lance_core::Result;
use lance_core::cache::{CacheCodecImpl, CacheEntryReader, CacheEntryWriter};
use lance_core::deepsize::DeepSizeOf;
use lance_core::utils::address::RowAddress;
use lance_select::{NullableRowAddrSet, RowAddrTreeMap, RowSetOps};
use roaring::RoaringBitmap;
use tracing::instrument;

use datafusion_common::ScalarValue;

use crate::metrics::MetricsCollector;
use crate::scalar::btree::{BTREE_VALUES_COLUMN, OrderableScalarValue};
use crate::scalar::{AnyQuery, SargableQuery};

const VALUES_COL_IDX: usize = 0;
const IDS_COL_IDX: usize = 1;

/// The rows of one page that matched a query, before they are assembled into
/// a [`NullableRowAddrSet`].
///
/// A btree search touches many pages, and each page's rows are scattered
/// across many fragments (a page holds rows sorted by *value*). Building a
/// `RowAddrTreeMap` per page and unioning them is O(pages x fragments) tiny
/// bitmaps; instead every page hands back its matched ids as plain arrays and
/// the caller assembles the final set once, e.g. with
/// [`RowAddrTreeMap::from_sorted_runs`].
///
/// Both arrays are sorted ascending because the page itself is sorted by row
/// id (see [`FlatIndex::try_new`]) and every array here is a filtered view of
/// the page's id column. `nulls` is empty unless the caller asked to track
/// nulls. A row id may appear in both (e.g. [`FlatIndex::all_matches`]), in
/// which case it is NULL — the same "null trumps true" rule as
/// [`NullableRowAddrSet::new`].
#[derive(Debug, Clone)]
pub struct PageMatches {
    selected: UInt64Array,
    nulls: UInt64Array,
}

impl PageMatches {
    fn new(selected: UInt64Array, nulls: UInt64Array) -> Self {
        Self { selected, nulls }
    }

    pub fn empty() -> Self {
        Self::new(empty_ids(), empty_ids())
    }

    /// Row ids for which the predicate was TRUE, sorted ascending.
    pub fn selected(&self) -> &[u64] {
        self.selected.values()
    }

    /// Row ids for which the predicate was NULL, sorted ascending.
    pub fn nulls(&self) -> &[u64] {
        self.nulls.values()
    }

    /// Build the set for this page alone. For many pages, assemble once from
    /// [`Self::selected`] / [`Self::nulls`] across all of them instead.
    pub fn into_row_addr_set(self) -> Result<NullableRowAddrSet> {
        Ok(NullableRowAddrSet::new(
            RowAddrTreeMap::from_sorted_iter(self.selected().iter().copied())?,
            RowAddrTreeMap::from_sorted_iter(self.nulls().iter().copied())?,
        ))
    }
}

fn empty_ids() -> UInt64Array {
    UInt64Array::from(Vec::<u64>::new())
}

/// A flat index is just a batch of value/row-id pairs
///
/// The batch always has two columns.  The first column "values" contains
/// the values.  The second column "row_ids" contains the row ids
///
/// Evaluating a query requires O(N) time where N is the # of rows
#[derive(Debug)]
pub struct FlatIndex {
    /// Sorted by row id. Nothing else is materialized at load time: every
    /// answer is a filtered view of the id column computed on demand, so a
    /// cached page costs exactly its Arrow buffers.
    data: Arc<RecordBatch>,
    df_schema: DFSchema,
}

impl DeepSizeOf for FlatIndex {
    fn deep_size_of_children(&self, _context: &mut lance_core::deepsize::Context) -> usize {
        // `df_schema` is a two-field schema derived from `data`; its footprint
        // is a few hundred bytes and not worth a separate accounting.
        self.data.get_array_memory_size()
    }
}

impl FlatIndex {
    #[instrument(name = "FlatIndex::try_new", level = "debug", skip_all)]
    pub fn try_new(data: RecordBatch) -> Result<Self> {
        // Sort by row id so every filtered view of the id column is itself
        // sorted, which is what makes bitmap construction cheap downstream.
        let data = data.sort_by_column(IDS_COL_IDX, None)?;
        let df_schema = DFSchema::try_from(data.schema())?;

        Ok(Self {
            data: Arc::new(data),
            df_schema,
        })
    }

    fn ids(&self) -> &UInt64Array {
        self.data.column(IDS_COL_IDX).as_primitive::<UInt64Type>()
    }

    fn values(&self) -> &ArrayRef {
        self.data.column(VALUES_COL_IDX)
    }

    fn has_nulls(&self) -> bool {
        self.values().null_count() > 0
    }

    /// The id column restricted to the rows where `mask` is true.
    fn filter_ids(&self, mask: &BooleanArray) -> Result<UInt64Array> {
        let filtered = arrow_select::filter::filter(self.ids(), mask)?;
        Ok(filtered
            .as_any()
            .downcast_ref::<UInt64Array>()
            .expect("Result of arrow_select::filter::filter did not match input type")
            .clone())
    }

    /// Row ids whose value is NULL.
    fn null_ids(&self) -> Result<UInt64Array> {
        if !self.has_nulls() {
            return Ok(empty_ids());
        }
        self.filter_ids(&arrow::compute::is_null(self.values())?)
    }

    /// Row ids whose value is not NULL.
    fn non_null_ids(&self) -> Result<UInt64Array> {
        if !self.has_nulls() {
            return Ok(self.ids().clone());
        }
        self.filter_ids(&arrow::compute::is_not_null(self.values())?)
    }

    /// Which of `needles` are present in this page.
    ///
    /// Batched existence sibling of [`Self::search`]: it runs the same `IsIn`
    /// predicate over the page's `values` column, but returns the matched
    /// *values* rather than row addresses — so the caller can map each result
    /// back to the input key it asked about. The page scan stays vectorized;
    /// only the (small) matched subset is lifted into `ScalarValue`.
    ///
    /// Nulls: a null `values` entry never matches a (non-null) primary-key
    /// needle, so it is simply absent from the result.
    pub(crate) fn contains_values(
        &self,
        needles: &[OrderableScalarValue],
    ) -> Result<BTreeSet<OrderableScalarValue>> {
        if needles.is_empty() {
            return Ok(BTreeSet::new());
        }
        let query = SargableQuery::IsIn(needles.iter().map(|v| v.0.clone()).collect());
        let expr = query.to_expr(BTREE_VALUES_COLUMN.to_string());
        let expr = create_physical_expr(&expr, &self.df_schema, &ExecutionProps::default())?;
        let predicate = expr.evaluate(&self.data)?;
        let predicate = predicate.into_array(self.data.num_rows())?;
        let predicate = predicate
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("Predicate should return boolean array");
        let matched = arrow_select::filter::filter(self.values(), predicate)?;
        (0..matched.len())
            .map(|i| {
                Ok(OrderableScalarValue(ScalarValue::try_from_array(
                    &matched, i,
                )?))
            })
            .collect()
    }

    /// Every row as TRUE, with the NULL rows also reported as NULL.
    pub fn all_matches(&self) -> Result<PageMatches> {
        // Some rows will be in both sets but that is ok, null trumps true
        Ok(PageMatches::new(self.ids().clone(), self.null_ids()?))
    }

    /// Every row as TRUE, NULL rows included, without reporting any NULLs.
    pub fn all_ignore_nulls_matches(&self) -> PageMatches {
        PageMatches::new(self.ids().clone(), empty_ids())
    }

    /// Every non-null row as TRUE without preserving NULL rows.
    pub fn all_non_null_matches(&self) -> Result<PageMatches> {
        Ok(PageMatches::new(self.non_null_ids()?, empty_ids()))
    }

    pub fn all(&self) -> Result<NullableRowAddrSet> {
        self.all_matches()?.into_row_addr_set()
    }

    pub fn all_ignore_nulls(&self) -> Result<NullableRowAddrSet> {
        self.all_ignore_nulls_matches().into_row_addr_set()
    }

    /// Return every non-null row as TRUE without preserving NULL rows.
    pub fn all_non_null(&self) -> Result<NullableRowAddrSet> {
        self.all_non_null_matches()?.into_row_addr_set()
    }

    pub fn remap_batch(batch: RecordBatch, mapping: &RowAddrRemap) -> Result<RecordBatch> {
        let row_ids = batch.column(IDS_COL_IDX).as_primitive::<UInt64Type>();
        let val_idx_and_new_id = row_ids
            .values()
            .iter()
            .enumerate()
            .filter_map(|(idx, old_id)| {
                mapping
                    .get(*old_id)
                    .unwrap_or(Some(*old_id))
                    .map(|new_id| (idx, new_id))
            })
            .collect::<Vec<_>>();
        let new_ids = Arc::new(UInt64Array::from_iter_values(
            val_idx_and_new_id.iter().copied().map(|(_, new_id)| new_id),
        ));
        let new_val_indices = UInt64Array::from_iter_values(
            val_idx_and_new_id
                .into_iter()
                .map(|(val_idx, _)| val_idx as u64),
        );
        let new_vals =
            arrow_select::take::take(batch.column(VALUES_COL_IDX), &new_val_indices, None)?;
        Ok(RecordBatch::try_new(
            batch.schema(),
            vec![new_vals, new_ids],
        )?)
    }

    /// Evaluate `query` against this page, returning the matched row ids.
    ///
    /// This is the per-page primitive; [`Self::search`] wraps it into a
    /// [`NullableRowAddrSet`] for callers that only look at one page.
    pub fn search_matches(
        &self,
        query: &dyn AnyQuery,
        track_nulls: bool,
        metrics: &dyn MetricsCollector,
    ) -> Result<PageMatches> {
        metrics.record_comparisons(self.data.num_rows());
        let query = query.as_any().downcast_ref::<SargableQuery>().unwrap();
        // Since we have all the values in memory we can use basic arrow-rs compute
        // functions to satisfy scalar queries.

        // Comparing anything with NULL yields NULL, so a NULL operand turns
        // every row of the page into NULL (or nothing, if nulls are not tracked).
        let everything_is_null = |this: &Self| {
            if track_nulls {
                PageMatches::new(empty_ids(), this.ids().clone())
            } else {
                PageMatches::empty()
            }
        };

        // Shortcuts for simple cases where we can re-use computed values
        match query {
            // x = NULL means all rows are NULL
            SargableQuery::Equals(value) => {
                if value.is_null() {
                    // if we have x = NULL then the correct SQL behavior is to return all NULLs
                    return Ok(everything_is_null(self));
                }
            }
            // x IS NULL is a filter on the values' validity, no predicate needed
            SargableQuery::IsNull() => {
                return Ok(PageMatches::new(self.null_ids()?, empty_ids()));
            }
            // x < NULL or x > NULL means all rows are NULL
            SargableQuery::Range(lower_bound, upper_bound) => match (lower_bound, upper_bound) {
                (Bound::Unbounded, Bound::Unbounded) => {
                    return Ok(self.all_ignore_nulls_matches());
                }
                (Bound::Unbounded, Bound::Included(upper) | Bound::Excluded(upper)) => {
                    if upper.is_null() {
                        return Ok(everything_is_null(self));
                    }
                }
                (Bound::Included(lower) | Bound::Excluded(lower), Bound::Unbounded)
                    if lower.is_null() =>
                {
                    return Ok(everything_is_null(self));
                }
                _ => {}
            },
            _ => {}
        };

        // No shortcut possible, need to actually evaluate the query
        let expr = query.to_expr(BTREE_VALUES_COLUMN.to_string());
        let expr = create_physical_expr(&expr, &self.df_schema, &ExecutionProps::default())?;
        self.eval_expr(&expr, track_nulls)
    }

    pub fn search(
        &self,
        query: &dyn AnyQuery,
        track_nulls: bool,
        metrics: &dyn MetricsCollector,
    ) -> Result<NullableRowAddrSet> {
        self.search_matches(query, track_nulls, metrics)?
            .into_row_addr_set()
    }

    /// Evaluate a predicate compiled once by the caller. Lets a large IsIn that
    /// spans many pages build the physical expr a single time instead of
    /// rebuilding the whole IN-list per page (the dominant cost of a big lookup).
    pub fn search_prebuilt_matches(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        track_nulls: bool,
        metrics: &dyn MetricsCollector,
    ) -> Result<PageMatches> {
        metrics.record_comparisons(self.data.num_rows());
        self.eval_expr(expr, track_nulls)
    }

    /// [`Self::search_prebuilt_matches`] wrapped into a [`NullableRowAddrSet`].
    pub fn search_prebuilt(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        track_nulls: bool,
        metrics: &dyn MetricsCollector,
    ) -> Result<NullableRowAddrSet> {
        self.search_prebuilt_matches(expr, track_nulls, metrics)?
            .into_row_addr_set()
    }

    fn eval_expr(&self, expr: &Arc<dyn PhysicalExpr>, track_nulls: bool) -> Result<PageMatches> {
        let predicate = expr.evaluate(&self.data)?;
        let predicate = predicate.into_array(self.data.num_rows())?;
        let predicate = predicate
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("Predicate should return boolean array");

        let selected = self.filter_ids(predicate)?;

        if !track_nulls {
            return Ok(PageMatches::new(selected, empty_ids()));
        }

        let nulls = if predicate.null_count() == 0 {
            empty_ids()
        } else {
            self.filter_ids(&arrow::compute::is_null(&predicate)?)?
        };

        Ok(PageMatches::new(selected, nulls))
    }

    pub fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        let mut frag_ids = self
            .ids()
            .values()
            .iter()
            .map(|row_id| RowAddress::from(*row_id).fragment_id())
            .collect::<Vec<_>>();
        frag_ids.sort();
        frag_ids.dedup();
        Ok(RoaringBitmap::from_sorted_iter(frag_ids).unwrap())
    }
}

impl CacheCodecImpl for FlatIndex {
    const TYPE_ID: &'static str = "lance.scalar.FlatIndex";
    /// v2 is the data batch alone. Entries written with the earlier layout
    /// (two roaring blobs ahead of the batch) fail to decode and are treated as
    /// cache misses, so the page is simply re-read from the index file.
    const CURRENT_VERSION: u32 = 2;

    fn serialize(&self, w: &mut CacheEntryWriter<'_>) -> Result<()> {
        // Format (v2):
        // ARROW_IPC : data batch
        w.write_ipc(self.data.as_ref())?;
        Ok(())
    }

    fn deserialize(r: &mut CacheEntryReader<'_>) -> Result<Self>
    where
        Self: Sized,
    {
        let batch = r.read_ipc()?;

        let df_schema = DFSchema::try_from(batch.schema())?;

        Ok(Self {
            data: Arc::new(batch),
            df_schema,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        metrics::NoOpMetricsCollector,
        scalar::btree::{BTREE_IDS_COLUMN, BTREE_VALUES_COLUMN},
    };

    use super::*;
    use arrow_array::{record_batch, types::Int32Type};
    use datafusion_common::ScalarValue;
    use lance_core::utils::row_addr_remap::GroupInput;
    use lance_datagen::{RowCount, array, gen_batch};
    use roaring::RoaringTreemap;
    use rstest::rstest;
    use std::collections::HashMap;

    fn example_index() -> FlatIndex {
        let batch = gen_batch()
            .col(
                "values",
                array::cycle::<Int32Type>(vec![10, 100, 1000, 1234]),
            )
            .col("ids", array::cycle::<UInt64Type>(vec![5, 0, 3, 100]))
            .into_batch_rows(RowCount::from(4))
            .unwrap();

        FlatIndex::try_new(batch).unwrap()
    }

    async fn check_index(query: &SargableQuery, expected: &[u64]) {
        let index = example_index();
        let actual = index.search(query, true, &NoOpMetricsCollector).unwrap();
        let expected =
            NullableRowAddrSet::new(RowAddrTreeMap::from_iter(expected), Default::default());
        assert_eq!(actual, expected);
    }

    fn assert_roundtrips(index: &FlatIndex) {
        let mut buf = Vec::new();
        index
            .serialize(&mut CacheEntryWriter::new(&mut buf))
            .unwrap();
        let data = bytes::Bytes::from(buf);
        let mut reader = CacheEntryReader::new(&data, 0, FlatIndex::CURRENT_VERSION);
        let restored = FlatIndex::deserialize(&mut reader).unwrap();

        assert_eq!(restored.data, index.data);
        assert_eq!(restored.all().unwrap(), index.all().unwrap());
    }

    /// The per-page primitives hand back sorted, filtered views of the id
    /// column; `all_matches` keeps NULL rows in `selected` (null trumps true),
    /// the other two drop or keep them as their names say.
    #[test]
    fn test_page_matches_views() {
        // ids are deliberately unsorted on input; try_new sorts by id.
        let batch = record_batch!(
            (
                BTREE_VALUES_COLUMN,
                Int32,
                [Some(3), None, Some(1), None, Some(2)]
            ),
            (BTREE_IDS_COLUMN, UInt64, [40, 10, 30, 50, 20])
        )
        .unwrap();
        let index = FlatIndex::try_new(batch).unwrap();

        let all = index.all_matches().unwrap();
        assert_eq!(all.selected(), &[10, 20, 30, 40, 50]);
        assert_eq!(all.nulls(), &[10, 50]);
        assert_eq!(
            all.into_row_addr_set().unwrap(),
            NullableRowAddrSet::new(
                RowAddrTreeMap::from_iter([10, 20, 30, 40, 50]),
                RowAddrTreeMap::from_iter([10, 50])
            )
        );

        let ignore = index.all_ignore_nulls_matches();
        assert_eq!(ignore.selected(), &[10, 20, 30, 40, 50]);
        assert!(ignore.nulls().is_empty());

        let non_null = index.all_non_null_matches().unwrap();
        assert_eq!(non_null.selected(), &[20, 30, 40]);
        assert!(non_null.nulls().is_empty());

        // Predicate path, with and without null tracking.
        let query = SargableQuery::Range(Bound::Included(ScalarValue::from(2)), Bound::Unbounded);
        let tracked = index
            .search_matches(&query, true, &NoOpMetricsCollector)
            .unwrap();
        assert_eq!(tracked.selected(), &[20, 40]);
        assert_eq!(tracked.nulls(), &[10, 50]);
        let untracked = index
            .search_matches(&query, false, &NoOpMetricsCollector)
            .unwrap();
        assert_eq!(untracked.selected(), &[20, 40]);
        assert!(untracked.nulls().is_empty());

        // A page without nulls answers the null shortcuts without scanning.
        let no_nulls = example_index();
        assert!(no_nulls.all_matches().unwrap().nulls().is_empty());
        assert_eq!(
            no_nulls.all_non_null_matches().unwrap().selected(),
            no_nulls.all_ignore_nulls_matches().selected()
        );
        assert!(
            no_nulls
                .search_matches(&SargableQuery::IsNull(), true, &NoOpMetricsCollector)
                .unwrap()
                .selected()
                .is_empty()
        );
    }

    #[test]
    fn test_cache_codec_roundtrip() {
        // No nulls
        assert_roundtrips(&example_index());

        // With nulls in the values column
        let batch = record_batch!(
            (BTREE_VALUES_COLUMN, Int32, [None, Some(0), Some(5)]),
            (BTREE_IDS_COLUMN, UInt64, [0, 1, 2])
        )
        .unwrap();
        assert_roundtrips(&FlatIndex::try_new(batch).unwrap());

        // Empty index
        let empty = RecordBatch::new_empty(example_index().data.schema());
        assert_roundtrips(&FlatIndex::try_new(empty).unwrap());
    }

    /// The data batch must decode zero-copy through the full envelope-bearing
    /// [`CacheCodec`], even though the envelope pushes the IPC section to a
    /// non-aligned starting offset.
    #[test]
    fn test_flat_index_data_is_zero_copy() {
        use lance_core::cache::CacheCodec;
        const ALIGN: usize = 64;

        let index = example_index();
        let codec = CacheCodec::from_impl::<FlatIndex>();
        let any: Arc<dyn std::any::Any + Send + Sync> = Arc::new(index);
        let mut buf = Vec::new();
        codec.serialize(&any, &mut buf).unwrap();

        let mut v = vec![0u8; buf.len() + ALIGN];
        let pad = (ALIGN - (v.as_ptr() as usize % ALIGN)) % ALIGN;
        v[pad..pad + buf.len()].copy_from_slice(&buf);
        let data = bytes::Bytes::from(v).slice(pad..pad + buf.len());

        let restored = codec.deserialize(&data).hit().unwrap();
        let restored = restored.downcast::<FlatIndex>().unwrap();

        let base = data.as_ptr() as usize;
        let end = base + data.len();
        for col in restored.data.columns() {
            for buffer in col.to_data().buffers() {
                let ptr = buffer.as_ptr() as usize;
                assert!(
                    ptr >= base && ptr < end,
                    "data batch buffer was realigned out of the input — misaligned IPC section",
                );
            }
        }
    }

    #[tokio::test]
    async fn test_equality() {
        check_index(&SargableQuery::Equals(ScalarValue::from(100)), &[0]).await;
        check_index(&SargableQuery::Equals(ScalarValue::from(10)), &[5]).await;
        check_index(&SargableQuery::Equals(ScalarValue::from(5)), &[]).await;
    }

    #[tokio::test]
    async fn test_range() {
        check_index(
            &SargableQuery::Range(
                Bound::Included(ScalarValue::from(100)),
                Bound::Excluded(ScalarValue::from(1234)),
            ),
            &[0, 3],
        )
        .await;
        check_index(
            &SargableQuery::Range(Bound::Unbounded, Bound::Excluded(ScalarValue::from(1000))),
            &[5, 0],
        )
        .await;
        check_index(
            &SargableQuery::Range(Bound::Included(ScalarValue::from(0)), Bound::Unbounded),
            &[5, 0, 3, 100],
        )
        .await;
        check_index(
            &SargableQuery::Range(Bound::Included(ScalarValue::from(100000)), Bound::Unbounded),
            &[],
        )
        .await;
    }

    #[tokio::test]
    async fn test_is_in() {
        check_index(
            &SargableQuery::IsIn(vec![
                ScalarValue::from(100),
                ScalarValue::from(1234),
                ScalarValue::from(3000),
            ]),
            &[0, 100],
        )
        .await;
    }

    #[tokio::test]
    async fn test_remap() {
        let index = example_index();
        // 0 -> 2000
        // 3 -> delete
        // Keep remaining as is
        let mapping = HashMap::<u64, Option<u64>>::from_iter(vec![(0, Some(2000)), (3, None)]);
        let remapped = FlatIndex::try_new(
            FlatIndex::remap_batch((*index.data).clone(), &RowAddrRemap::direct(mapping)).unwrap(),
        )
        .unwrap();

        let expected = FlatIndex::try_new(
            gen_batch()
                .col("values", array::cycle::<Int32Type>(vec![10, 100, 1234]))
                .col("ids", array::cycle::<UInt64Type>(vec![5, 2000, 100]))
                .into_batch_rows(RowCount::from(3))
                .unwrap(),
        )
        .unwrap();
        assert_eq!(remapped.data, expected.data);
    }

    // An entire page (frag 0) is deleted during compaction. remap_batch must
    // drop every row regardless of which RowAddrRemap mode expresses it.
    // example_index holds row ids 5, 0, 3, 100, all in frag 0.
    #[rstest]
    #[case::compact(RowAddrRemap::compact([GroupInput {
        rewritten_old_row_addrs: RoaringTreemap::new(),
        old_frag_ids: vec![0],
        new_frags: vec![],
    }])
    .unwrap())]
    #[case::explicit(RowAddrRemap::direct(
        [5u64, 0, 3, 100].into_iter().map(|id| (id, None)).collect(),
    ))]
    fn test_remap_to_nothing(#[case] remap: RowAddrRemap) {
        let index = example_index();
        let remapped = FlatIndex::remap_batch((*index.data).clone(), &remap).unwrap();
        assert_eq!(remapped.num_rows(), 0);
    }

    #[test]
    fn test_null_handling() {
        // [null, 0, 5]
        let batch = record_batch!(
            (BTREE_VALUES_COLUMN, Int32, [None, Some(0), Some(5)]),
            (BTREE_IDS_COLUMN, UInt64, [0, 1, 2])
        )
        .unwrap();
        let index = FlatIndex::try_new(batch).unwrap();

        let check = |query: SargableQuery, true_ids: &[u64], null_ids: &[u64]| {
            let actual = index.search(&query, true, &NoOpMetricsCollector).unwrap();
            let expected = NullableRowAddrSet::new(
                RowAddrTreeMap::from_iter(true_ids),
                RowAddrTreeMap::from_iter(null_ids),
            );
            assert_eq!(actual, expected, "query: {:?}", query);
        };

        let null = ScalarValue::Int32(None);
        let zero = ScalarValue::Int32(Some(0));
        let three = ScalarValue::Int32(Some(3));

        check(SargableQuery::Equals(zero.clone()), &[1], &[0]);
        // x = NULL returns all rows as NULL and nothing as TRUE
        check(SargableQuery::Equals(null.clone()), &[], &[0, 1, 2]);

        check(SargableQuery::IsIn(vec![zero.clone()]), &[1], &[0]);
        // x IN (0, NULL) promotes all FALSE to NULL
        check(SargableQuery::IsIn(vec![zero, null.clone()]), &[1], &[0, 2]);

        check(SargableQuery::IsNull(), &[0], &[]);

        check(
            SargableQuery::Range(Bound::Included(three.clone()), Bound::Unbounded),
            &[2],
            &[0],
        );

        // x < NULL or x > NULL returns everything as NULL
        check(
            SargableQuery::Range(Bound::Unbounded, Bound::Included(null.clone())),
            &[],
            &[0, 1, 2],
        );

        check(
            SargableQuery::Range(Bound::Excluded(null.clone()), Bound::Unbounded),
            &[],
            &[0, 1, 2],
        );

        // x BETWEEN 3 AND NULL returns everything as NULL unless we know it is FALSE
        check(
            SargableQuery::Range(
                Bound::Included(three.clone()),
                Bound::Included(null.clone()),
            ),
            &[],
            &[0, 2],
        );
        check(
            SargableQuery::Range(Bound::Included(null.clone()), Bound::Included(three)),
            &[],
            &[0, 1],
        );
        check(
            SargableQuery::Range(Bound::Included(null.clone()), Bound::Included(null)),
            &[],
            &[0, 1, 2],
        );
    }

    /// Row addresses pack `(fragment_id << 32) | offset`, so a page spanning
    /// several fragments must report exactly those fragments, deduped and
    /// sorted. Every other test in this module uses offsets inside fragment 0,
    /// which never exercises the shift.
    #[test]
    fn test_calculate_included_frags_spans_fragments() {
        let addr = |frag: u32, offset: u32| u64::from(RowAddress::new_from_parts(frag, offset));
        let batch = record_batch!(
            (
                BTREE_VALUES_COLUMN,
                Int32,
                [Some(1), Some(2), Some(3), Some(4)]
            ),
            (
                BTREE_IDS_COLUMN,
                UInt64,
                [addr(0, 0), addr(2, 7), addr(0, 1), addr(5, 3)]
            )
        )
        .unwrap();
        let index = FlatIndex::try_new(batch).unwrap();

        assert_eq!(
            index.calculate_included_frags().unwrap(),
            RoaringBitmap::from_iter([0u32, 2, 5])
        );

        // A hit still carries the full 64-bit address, not a bare offset.
        let hit = index
            .search(
                &SargableQuery::Equals(ScalarValue::from(2)),
                true,
                &NoOpMetricsCollector,
            )
            .unwrap();
        assert_eq!(
            hit,
            NullableRowAddrSet::new(RowAddrTreeMap::from_iter(&[addr(2, 7)]), Default::default())
        );
    }

    /// A zero-row page has to answer queries with empty sets rather than
    /// panicking inside the Arrow predicate evaluation. The roundtrip test
    /// builds an empty index but never queries one. Both `track_nulls` modes
    /// take different shortcuts, and neither has any row to return here.
    #[test]
    fn test_empty_index_answers_queries_with_empty_sets() {
        let empty = RecordBatch::new_empty(example_index().data.schema());
        let index = FlatIndex::try_new(empty).unwrap();
        let nothing = NullableRowAddrSet::new(RowAddrTreeMap::new(), RowAddrTreeMap::new());

        for query in [
            SargableQuery::Equals(ScalarValue::from(10)),
            SargableQuery::Equals(ScalarValue::Int32(None)),
            SargableQuery::IsNull(),
            SargableQuery::IsIn(vec![ScalarValue::from(10), ScalarValue::from(20)]),
            SargableQuery::Range(Bound::Unbounded, Bound::Unbounded),
        ] {
            for track_nulls in [true, false] {
                assert_eq!(
                    index
                        .search(&query, track_nulls, &NoOpMetricsCollector)
                        .unwrap(),
                    nothing,
                    "query: {query:?}, track_nulls: {track_nulls}"
                );
            }
        }

        assert!(index.all().unwrap().true_rows().is_empty());
        assert_eq!(
            index.calculate_included_frags().unwrap(),
            RoaringBitmap::new()
        );
    }
}
