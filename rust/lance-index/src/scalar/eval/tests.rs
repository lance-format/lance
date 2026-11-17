// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::any::Any;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use datafusion_common::ScalarValue;
use futures::future::BoxFuture;
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::utils::row_addr_remap::RowAddrRemap;
use lance_select::{RowAddrMask, RowAddrTreeMap};
use roaring::RoaringBitmap;
use tokio::sync::{Barrier, Notify};

use super::*;
use crate::metrics::NoOpMetricsCollector;
use crate::scalar::{
    AnyQuery, CreatedIndex, FullTextSearchQuery, IndexStore, LabelListQuery, ScalarIndex,
    ScalarIndexParams, TokenQuery, UpdateCriteria,
};
use crate::{Index, IndexType};

type LoadHook = Arc<dyn Fn(&str) -> BoxFuture<'static, ()> + Send + Sync>;

struct InFlightGuard(Arc<AtomicUsize>);

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

struct DropCounterGuard(Arc<AtomicUsize>);

impl Drop for DropCounterGuard {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[derive(Debug)]
struct MockStats {
    load_calls: AtomicUsize,
    search_calls: AtomicUsize,
    remap_calls: AtomicUsize,
    in_flight: AtomicUsize,
    max_in_flight: AtomicUsize,
    search_order: Mutex<Vec<String>>,
}

impl MockStats {
    fn new() -> Self {
        Self {
            load_calls: AtomicUsize::new(0),
            search_calls: AtomicUsize::new(0),
            remap_calls: AtomicUsize::new(0),
            in_flight: AtomicUsize::new(0),
            max_in_flight: AtomicUsize::new(0),
            search_order: Mutex::new(Vec::new()),
        }
    }
}

#[derive(Clone, Debug)]
enum MockOutcome {
    Exact { selected: Vec<u64>, nulls: Vec<u64> },
    AtMost { selected: Vec<u64>, nulls: Vec<u64> },
    AtLeast { selected: Vec<u64>, nulls: Vec<u64> },
}

impl MockOutcome {
    fn to_search_result(&self, track_nulls: bool) -> SearchResult {
        let make_set = |selected: &[u64], nulls: &[u64]| {
            if track_nulls {
                NullableRowAddrSet::new(
                    RowAddrTreeMap::from_iter(selected.iter().copied()),
                    RowAddrTreeMap::from_iter(nulls.iter().copied()),
                )
            } else {
                // Simulate an index omitting NULL rows when `track_nulls == false`.
                let nulls_map = RowAddrTreeMap::from_iter(nulls.iter().copied());
                let true_map = RowAddrTreeMap::from_iter(selected.iter().copied()) - &nulls_map;
                NullableRowAddrSet::new(true_map, RowAddrTreeMap::new())
            }
        };
        match self {
            Self::Exact { selected, nulls } => SearchResult::Exact(make_set(selected, nulls)),
            Self::AtMost { selected, nulls } => SearchResult::AtMost(make_set(selected, nulls)),
            Self::AtLeast { selected, nulls } => SearchResult::AtLeast(make_set(selected, nulls)),
        }
    }
}

struct MockIndex {
    column: String,
    stats: Arc<MockStats>,
    outcomes: HashMap<String, MockOutcome>,
    yield_on_search: bool,
    row_addresses: bool,
    on_search: Option<LoadHook>,
}

impl std::fmt::Debug for MockIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockIndex")
            .field("column", &self.column)
            .field("row_addresses", &self.row_addresses)
            .finish()
    }
}

impl DeepSizeOf for MockIndex {
    fn deep_size_of_children(&self, _context: &mut Context) -> usize {
        0
    }
}

#[async_trait]
impl Index for MockIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }
    fn statistics(&self) -> Result<serde_json::Value> {
        Ok(serde_json::Value::Null)
    }
    async fn prewarm(&self) -> Result<()> {
        Ok(())
    }
    fn index_type(&self) -> IndexType {
        IndexType::Scalar
    }
    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        Ok(RoaringBitmap::new())
    }
}

#[async_trait]
impl ScalarIndex for MockIndex {
    async fn search(&self, query: &dyn AnyQuery, _: &dyn MetricsCollector) -> Result<SearchResult> {
        self.search_with_options(query, SearchOptions::default(), &NoOpMetricsCollector)
            .await
    }

    async fn search_with_options(
        &self,
        query: &dyn AnyQuery,
        options: SearchOptions,
        _: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        struct StatsInFlightGuard<'a>(&'a MockStats);
        impl Drop for StatsInFlightGuard<'_> {
            fn drop(&mut self) {
                self.0.in_flight.fetch_sub(1, Ordering::SeqCst);
            }
        }

        self.stats.search_calls.fetch_add(1, Ordering::SeqCst);
        let cur = self.stats.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.stats.max_in_flight.fetch_max(cur, Ordering::SeqCst);
        let _in_flight_guard = StatsInFlightGuard(&self.stats);

        let key = query.format(&self.column);
        self.stats.search_order.lock().unwrap().push(key.clone());

        if let Some(on_search) = &self.on_search {
            on_search(&key).await;
        }
        if self.yield_on_search {
            tokio::task::yield_now().await;
        }

        let outcome = self
            .outcomes
            .get(&key)
            .cloned()
            .unwrap_or(MockOutcome::Exact {
                selected: vec![],
                nulls: vec![],
            });
        Ok(outcome.to_search_result(options.track_nulls()))
    }

    fn results_are_row_addresses(&self) -> bool {
        self.row_addresses
    }

    async fn remap(&self, _: &RowAddrRemap, _: &dyn IndexStore) -> Result<CreatedIndex> {
        Err(Error::not_supported("mock"))
    }
    async fn update(
        &self,
        _: datafusion::execution::SendableRecordBatchStream,
        _: &dyn IndexStore,
        _: Option<crate::scalar::OldIndexDataFilter>,
    ) -> Result<CreatedIndex> {
        Err(Error::not_supported("mock"))
    }
    fn update_criteria(&self) -> UpdateCriteria {
        UpdateCriteria::only_new_data(crate::scalar::TrainingCriteria::new(
            crate::scalar::TrainingOrdering::None,
        ))
    }
    fn derive_index_params(&self) -> Result<ScalarIndexParams> {
        Err(Error::not_supported("mock"))
    }
}

struct MockLoader {
    stats: Arc<MockStats>,
    outcomes: HashMap<String, MockOutcome>,
    yield_on_search: bool,
    row_addresses: bool,
    row_id_columns: HashMap<String, bool>,
    on_load: Option<LoadHook>,
    on_search: Option<LoadHook>,
    yield_on_remap: bool,
}

impl MockLoader {
    fn new(outcomes: HashMap<String, MockOutcome>) -> Self {
        Self {
            stats: Arc::new(MockStats::new()),
            outcomes,
            yield_on_search: false,
            row_addresses: true,
            row_id_columns: HashMap::new(),
            on_load: None,
            on_search: None,
            yield_on_remap: false,
        }
    }
}

#[async_trait]
impl ScalarIndexLoader for MockLoader {
    async fn load_index(
        &self,
        column: &str,
        _index_name: &str,
        _metrics: &dyn MetricsCollector,
    ) -> Result<Arc<dyn ScalarIndex>> {
        self.stats.load_calls.fetch_add(1, Ordering::SeqCst);
        if let Some(on_load) = &self.on_load {
            on_load(column).await;
        }
        let row_addresses = if self.row_id_columns.contains_key(column) {
            false
        } else {
            self.row_addresses
        };
        Ok(Arc::new(MockIndex {
            column: column.to_string(),
            stats: Arc::clone(&self.stats),
            outcomes: self.outcomes.clone(),
            yield_on_search: self.yield_on_search,
            row_addresses,
            on_search: self.on_search.clone(),
        }))
    }

    async fn row_id_result_to_row_addrs(
        &self,
        result: NullableIndexExprResult,
    ) -> Result<NullableIndexExprResult> {
        self.stats.remap_calls.fetch_add(1, Ordering::SeqCst);
        if self.yield_on_remap {
            tokio::task::yield_now().await;
            tokio::task::yield_now().await;
        }
        Ok(result.with_domain(MaskDomain::RowAddress))
    }
}

fn q_search(
    column: &str,
    index_type: &str,
    query: impl AnyQuery,
    needs_recheck: bool,
) -> ScalarIndexExpr {
    ScalarIndexExpr::Query(ScalarIndexSearch {
        column: column.to_string(),
        index_name: format!("{column}_idx"),
        index_type: index_type.to_string(),
        query: Arc::new(query),
        needs_recheck,
        fragment_bitmap: None,
    })
}

#[async_recursion]
async fn reference_eval_nullable(
    expr: &ScalarIndexExpr,
    loader: &dyn ScalarIndexLoader,
    metrics: &dyn MetricsCollector,
    track_nulls: bool,
) -> Result<NullableIndexExprResult> {
    match expr {
        ScalarIndexExpr::Not(inner) => {
            Ok(!reference_eval_nullable(inner, loader, metrics, true).await?)
        }
        ScalarIndexExpr::And(lhs, rhs) => {
            let l = reference_eval_nullable(lhs, loader, metrics, track_nulls).await?;
            let r = reference_eval_nullable(rhs, loader, metrics, track_nulls).await?;
            let (l, r) = match (l.domain, r.domain) {
                (MaskDomain::RowId, MaskDomain::RowAddress) => {
                    (loader.row_id_result_to_row_addrs(l).await?, r)
                }
                (MaskDomain::RowAddress, MaskDomain::RowId) => {
                    (l, loader.row_id_result_to_row_addrs(r).await?)
                }
                _ => (l, r),
            };
            Ok(l & r)
        }
        ScalarIndexExpr::Or(lhs, rhs) => {
            let l = reference_eval_nullable(lhs, loader, metrics, track_nulls).await?;
            let r = reference_eval_nullable(rhs, loader, metrics, track_nulls).await?;
            let (l, r) = match (l.domain, r.domain) {
                (MaskDomain::RowId, MaskDomain::RowAddress) => {
                    (loader.row_id_result_to_row_addrs(l).await?, r)
                }
                (MaskDomain::RowAddress, MaskDomain::RowId) => {
                    (l, loader.row_id_result_to_row_addrs(r).await?)
                }
                _ => (l, r),
            };
            Ok(l | r)
        }
        ScalarIndexExpr::Query(search) => {
            let res = eval_search(search, loader, metrics, track_nulls).await?;
            Ok(res)
        }
    }
}

#[tokio::test]
async fn test_cse_dedup_counts_on_256_branch_composite_or() {
    // 256-branch OR where each branch is:
    //   grp = 'hot_grp'|'cold_grp' AND item_id = i AND ts >= 100 AND ts <= 200
    // After optimize(), ts merges into Range(100, 200) -> 256 occurrences.
    // grp = 'hot_grp' has 128 occurrences, grp = 'cold_grp' has 128 occurrences.
    // item_id = i has 1 occurrence each (256 unique leaves, must NOT be cached in slots).
    let mut outcomes = HashMap::new();
    outcomes.insert(
        "grp = hot_grp".to_string(),
        MockOutcome::Exact {
            selected: vec![1, 2, 3],
            nulls: vec![],
        },
    );
    outcomes.insert(
        "grp = cold_grp".to_string(),
        MockOutcome::Exact {
            selected: vec![4, 5],
            nulls: vec![],
        },
    );
    outcomes.insert(
        "ts >= 100 && ts <= 200".to_string(),
        MockOutcome::Exact {
            selected: vec![2, 4, 10],
            nulls: vec![],
        },
    );
    for i in 0..256_i64 {
        let selected = if i == 42 { vec![2] } else { vec![] };
        outcomes.insert(
            format!("item_id = {i}"),
            MockOutcome::Exact {
                selected,
                nulls: vec![],
            },
        );
    }

    let branches: Vec<ScalarIndexExpr> = (0..256_i64)
        .map(|i| {
            let grp_val = if i % 2 == 0 { "hot_grp" } else { "cold_grp" };
            let grp = q_search(
                "grp",
                "Bitmap",
                SargableQuery::Equals(ScalarValue::Utf8(Some(grp_val.to_string()))),
                false,
            );
            let item = q_search(
                "item_id",
                "BTree",
                SargableQuery::Equals(ScalarValue::Int64(Some(i))),
                false,
            );
            let ts_lo = q_search(
                "ts",
                "BTree",
                SargableQuery::Range(
                    Bound::Included(ScalarValue::Int64(Some(100))),
                    Bound::Unbounded,
                ),
                false,
            );
            let ts_hi = q_search(
                "ts",
                "BTree",
                SargableQuery::Range(
                    Bound::Unbounded,
                    Bound::Included(ScalarValue::Int64(Some(200))),
                ),
                false,
            );
            ScalarIndexExpr::And(
                Box::new(ScalarIndexExpr::And(Box::new(grp), Box::new(item))),
                Box::new(ScalarIndexExpr::And(Box::new(ts_lo), Box::new(ts_hi))),
            )
        })
        .collect();

    let expr = branches
        .into_iter()
        .reduce(|a, b| ScalarIndexExpr::Or(Box::new(a), Box::new(b)))
        .unwrap()
        .optimize();

    // Only 3 repeated canonical leaves exist (`hot_grp`, `cold_grp`, `ts in [100, 200]`).
    let ctx = EvalContext::new(&expr, false);
    assert_eq!(ctx.slots.len(), 3);

    // Verify unique `item_id` leaves bypass cache (`slot_id == None`).
    let unique_count = ctx
        .leaf_meta
        .values()
        .filter(|m| m.slot_id.is_none() && m.occurrence_count == 1)
        .count();
    assert_eq!(unique_count, 256);

    let loader = MockLoader::new(outcomes);
    let result = expr.evaluate(&loader, &NoOpMetricsCollector).await.unwrap();
    assert_eq!(
        result.lower,
        RowAddrMask::from_allowed([2].into_iter().collect())
    );
    assert!(result.is_exact());

    // Each of the 256 branches runs unique `item_id = i` FIRST (before repeated Bitmap `grp`
    // and `ts` Range). Only branch 42 returns non-empty for `item_id`, so only branch 42
    // evaluates `grp = 'hot_grp'` and `ts in [100, 200]`; `grp = 'cold_grp'` is never even
    // needed! Total calls = 256 (item_id) + 1 (hot_grp) + 1 (ts) = 258, with zero domain
    // conversions because all leaves are native row-address domain.
    assert_eq!(loader.stats.load_calls.load(Ordering::SeqCst), 258);
    assert_eq!(loader.stats.search_calls.load(Ordering::SeqCst), 258);
    assert_eq!(loader.stats.remap_calls.load(Ordering::SeqCst), 0);

    // Now force ALL 3 shared leaves to execute across multiple matching branches
    // (e.g. `item_id` matches in branches 0 and 1 so both `hot_grp` and `cold_grp` and `ts`
    // run across 256 branches): each shared leaf must STILL execute at most once!
    let mut outcomes2 = HashMap::new();
    outcomes2.insert(
        "grp = hot_grp".to_string(),
        MockOutcome::Exact {
            selected: vec![1, 2],
            nulls: vec![],
        },
    );
    outcomes2.insert(
        "grp = cold_grp".to_string(),
        MockOutcome::Exact {
            selected: vec![3, 4],
            nulls: vec![],
        },
    );
    outcomes2.insert(
        "ts >= 100 && ts <= 200".to_string(),
        MockOutcome::Exact {
            selected: vec![1, 3],
            nulls: vec![],
        },
    );
    for i in 0..256_i64 {
        outcomes2.insert(
            format!("item_id = {i}"),
            MockOutcome::Exact {
                selected: vec![1, 3],
                nulls: vec![],
            },
        );
    }
    let mut loader2 = MockLoader::new(outcomes2);
    // Mark `grp` as row-id domain while `item_id` and `ts` stay row-address domain:
    // CSE must memoize the row-id -> row-address conversion once per unique `grp` leaf
    // (`hot_grp` and `cold_grp`), never 256 times!
    loader2.row_id_columns.insert("grp".to_string(), true);
    let result2 = expr
        .evaluate(&loader2, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert_eq!(result2.domain, MaskDomain::RowAddress);
    assert_eq!(
        result2.lower,
        RowAddrMask::from_allowed([1, 3].into_iter().collect())
    );
    // 256 unique item_id + 1 hot_grp + 1 cold_grp + 1 ts = 259 (NEVER 768!),
    // and only 2 domain conversions (`hot_grp` + `cold_grp`, NEVER 256!).
    assert_eq!(loader2.stats.load_calls.load(Ordering::SeqCst), 259);
    assert_eq!(loader2.stats.search_calls.load(Ordering::SeqCst), 259);
    assert_eq!(loader2.stats.remap_calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn test_cse_separates_track_nulls_for_q_and_not_q() {
    // q: rows {1 (TRUE), 2 (NULL)}; row 0 is FALSE.
    // Under track_nulls=false, MockIndex drops row 2. Under track_nulls=true (NOT),
    // MockIndex preserves row 2 in `nulls`.
    // `q AND NOT(q)`:
    // - row 1: TRUE & FALSE = FALSE
    // - row 2: NULL & NULL = NULL -> dropped by WHERE => empty
    // `OR(q, NOT(q))` vs `AND(q, NOT(q))`: let's test `OR(AND(r, q), NOT(q))` too:
    // Wait: specifically test `AND(q, NOT(q))` AND `AND(q, NOT(q_other))` and ensure
    // `(q, false)` and `(q, true)` do NOT collide in CSE!
    let mut outcomes = HashMap::new();
    outcomes.insert(
        "x = 1".to_string(),
        MockOutcome::Exact {
            selected: vec![1, 2],
            nulls: vec![2],
        },
    );
    outcomes.insert(
        "y = 1".to_string(),
        MockOutcome::Exact {
            selected: vec![0, 1, 2],
            nulls: vec![],
        },
    );

    let q = q_search(
        "x",
        "BTree",
        SargableQuery::Equals(ScalarValue::Int64(Some(1))),
        false,
    );
    let not_q = ScalarIndexExpr::Not(Box::new(q.clone()));
    let y = q_search(
        "y",
        "BTree",
        SargableQuery::Equals(ScalarValue::Int64(Some(1))),
        false,
    );

    // Expression: (y AND NOT(x = 1)) AND (x = 1 OR y = 1)
    // Wait: let's check both `(x = 1) AND NOT(x = 1)` and `OR(x = 1, NOT(x = 1))`!
    let or_q_not_q = ScalarIndexExpr::Or(Box::new(q.clone()), Box::new(not_q.clone()));
    let ctx_or = EvalContext::new(&or_q_not_q, false);
    // `(x = 1, false)` appears once and `(x = 1, true)` appears once => 0 shared slots!
    assert_eq!(ctx_or.slots.len(), 0);

    let loader = MockLoader::new(outcomes.clone());
    let res = or_q_not_q
        .evaluate(&loader, &NoOpMetricsCollector)
        .await
        .unwrap();
    // Row 0: FALSE | TRUE = TRUE. Row 1: TRUE | FALSE = TRUE. Row 2: NULL | NULL = NULL -> dropped.
    assert!(res.lower.selected(0));
    assert!(res.lower.selected(1));
    assert!(!res.lower.selected(2)); // NULL must NOT turn into TRUE!
    assert_eq!(loader.stats.search_calls.load(Ordering::SeqCst), 2);

    // Also test `AND(q, NOT(q))` where `q` appears twice with `false` and twice with `true`:
    let dup_and = ScalarIndexExpr::And(
        Box::new(ScalarIndexExpr::And(Box::new(q.clone()), Box::new(q))),
        Box::new(ScalarIndexExpr::And(
            Box::new(not_q.clone()),
            Box::new(not_q),
        )),
    );
    let ctx_dup = EvalContext::new(&dup_and, false);
    assert_eq!(ctx_dup.slots.len(), 2); // One slot for track_nulls=false, one for track_nulls=true.

    let loader2 = MockLoader::new(outcomes);
    let res2 = dup_and
        .evaluate(&loader2, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert_eq!(res2.lower, RowAddrMask::allow_nothing());
    // Check `y` unused warning avoided via evaluating `(y AND NOT(q))`
    let y_and_not_q = ScalarIndexExpr::And(
        Box::new(y),
        Box::new(ScalarIndexExpr::Not(Box::new(q_search(
            "x",
            "BTree",
            SargableQuery::Equals(ScalarValue::Int64(Some(1))),
            false,
        )))),
    );
    let res3 = y_and_not_q
        .evaluate(&loader2, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert!(res3.lower.selected(0));
    assert!(!res3.lower.selected(1));
    assert!(!res3.lower.selected(2));
}

#[tokio::test]
async fn test_or_bounded_concurrency_128_branches() {
    let mut outcomes = HashMap::new();
    for i in 0..128_i64 {
        outcomes.insert(
            format!("x = {i}"),
            MockOutcome::Exact {
                selected: vec![i as u64],
                nulls: vec![],
            },
        );
    }
    let expr = (0..128_i64)
        .map(|i| {
            q_search(
                "x",
                "BTree",
                SargableQuery::Equals(ScalarValue::Int64(Some(i))),
                false,
            )
        })
        .reduce(|a, b| ScalarIndexExpr::Or(Box::new(a), Box::new(b)))
        .unwrap();

    let mut loader = MockLoader::new(outcomes);
    loader.yield_on_search = true;

    let res = expr.evaluate(&loader, &NoOpMetricsCollector).await.unwrap();
    let max_in_flight = loader.stats.max_in_flight.load(Ordering::SeqCst);
    assert!(
        max_in_flight > 1 && max_in_flight <= MAX_OR_CONCURRENCY,
        "expected 1 < max_in_flight <= {MAX_OR_CONCURRENCY}, got {max_in_flight}"
    );
    for i in 0..128_u64 {
        assert!(res.lower.selected(i));
    }
}

#[tokio::test]
async fn test_and_selective_ordering_and_short_circuiting() {
    // Case 1: Selective ordering: unique BTree Equals runs before repeated Bitmap Equals
    // and two-sided Range, and empty unique BTree Equals short-circuits the others.
    let mut outcomes = HashMap::new();
    outcomes.insert(
        "a = 1".to_string(),
        MockOutcome::Exact {
            selected: vec![],
            nulls: vec![],
        },
    );
    outcomes.insert(
        "b = 2".to_string(),
        MockOutcome::Exact {
            selected: vec![1, 2],
            nulls: vec![],
        },
    );
    outcomes.insert(
        "c >= 10 && c <= 20".to_string(),
        MockOutcome::AtMost {
            selected: vec![1, 2],
            nulls: vec![],
        },
    );

    let b_repeated = q_search(
        "b",
        "Bitmap",
        SargableQuery::Equals(ScalarValue::Int64(Some(2))),
        false,
    );
    let c_range = q_search(
        "c",
        "ZoneMap",
        SargableQuery::Range(
            Bound::Included(ScalarValue::Int64(Some(10))),
            Bound::Included(ScalarValue::Int64(Some(20))),
        ),
        true,
    );
    let a_unique = q_search(
        "a",
        "BTree",
        SargableQuery::Equals(ScalarValue::Int64(Some(1))),
        false,
    );

    // Place c_range and repeated b_repeated BEFORE a_unique in the AST;
    // Also add a second occurrence of b_repeated in an OR so b_repeated has count == 2.
    let and_branch = ScalarIndexExpr::And(
        Box::new(c_range),
        Box::new(ScalarIndexExpr::And(
            Box::new(b_repeated.clone()),
            Box::new(a_unique),
        )),
    );
    let expr = ScalarIndexExpr::Or(Box::new(and_branch), Box::new(b_repeated));

    let loader = MockLoader::new(outcomes.clone());
    let res = expr.evaluate(&loader, &NoOpMetricsCollector).await.unwrap();
    // Branch 1 evaluates a_unique FIRST and short-circuits c_range (needs_recheck=true),
    // so branch 1 yields AtMost(allow_nothing()); OR with b_repeated yields AtMost({1, 2}).
    assert!(!res.is_exact());
    assert_eq!(
        res.upper,
        RowAddrMask::from_allowed([1, 2].into_iter().collect())
    );
    let order = loader.stats.search_order.lock().unwrap().clone();
    assert!(order.contains(&"a = 1".to_string()));
    assert!(!order.contains(&"c >= 10 && c <= 20".to_string()));

    // Case 2: All skipped conjuncts exact => short-circuit stays Exact!
    let mut exact_outcomes = HashMap::new();
    exact_outcomes.insert(
        "a = 1".to_string(),
        MockOutcome::Exact {
            selected: vec![],
            nulls: vec![],
        },
    );
    exact_outcomes.insert(
        "b = 2".to_string(),
        MockOutcome::Exact {
            selected: vec![1],
            nulls: vec![],
        },
    );
    let exact_and = ScalarIndexExpr::And(
        Box::new(q_search(
            "b",
            "Bitmap",
            SargableQuery::Equals(ScalarValue::Int64(Some(2))),
            false,
        )),
        Box::new(q_search(
            "a",
            "BTree",
            SargableQuery::Equals(ScalarValue::Int64(Some(1))),
            false,
        )),
    );
    let loader_exact = MockLoader::new(exact_outcomes);
    let exact_res = exact_and
        .evaluate(&loader_exact, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert!(exact_res.is_exact());
    assert_eq!(exact_res.lower, RowAddrMask::allow_nothing());
    assert_eq!(
        loader_exact.stats.search_order.lock().unwrap().clone(),
        vec!["a = 1".to_string()]
    );

    // Case 3: NULL-only AllowList inside NOT must NOT short-circuit!
    // `NOT(a = 1 AND b = 2)` where row 5 is NULL for `a = 1` and FALSE for `b = 2`:
    // `NULL & FALSE = FALSE`, so `NOT(FALSE) = TRUE`!
    // If `a = 1` (returning `selected={5}, nulls={5}`) short-circuited `b = 2` and returned
    // `allow_nothing()`, `NOT` would treat row 5 as TRUE even if `b = 2` were TRUE (where
    // `NULL & TRUE = NULL`, `NOT(NULL) = NULL != TRUE`)!
    let mut null_outcomes = HashMap::new();
    null_outcomes.insert(
        "a = 1".to_string(),
        MockOutcome::Exact {
            selected: vec![5],
            nulls: vec![5],
        },
    );
    null_outcomes.insert(
        "b = 2".to_string(),
        MockOutcome::Exact {
            selected: vec![5],
            nulls: vec![],
        },
    );
    let not_and = ScalarIndexExpr::Not(Box::new(ScalarIndexExpr::And(
        Box::new(q_search(
            "a",
            "BTree",
            SargableQuery::Equals(ScalarValue::Int64(Some(1))),
            false,
        )),
        Box::new(q_search(
            "b",
            "Bitmap",
            SargableQuery::Equals(ScalarValue::Int64(Some(2))),
            false,
        )),
    )));
    let loader_null = MockLoader::new(null_outcomes);
    let res_null = not_and
        .evaluate(&loader_null, &NoOpMetricsCollector)
        .await
        .unwrap();
    // `b = 2` MUST have been evaluated (no short-circuit on NULL-only AllowList under NOT).
    assert_eq!(loader_null.stats.search_calls.load(Ordering::SeqCst), 2);
    // Row 5 is NULL & TRUE = NULL -> NOT(NULL) = NULL -> dropped (NOT selected!).
    assert!(!res_null.lower.selected(5));
    assert!(res_null.lower.selected(0));
}

#[tokio::test]
async fn test_conjunct_ordering_tiers() {
    // Verify full priority order when none short-circuit:
    // 1. ready CSE
    // 2. unique BTree Equals < unique Bitmap Equals < repeated BTree Equals < repeated Bitmap Equals
    // 3. narrow IsIn (<= 16): BTree len 2 < Bitmap len 2 < BTree len 4
    // 4. other queries: large IsIn (>16) < LikePrefix < 2-sided Range < 1-sided Range < FTS/LabelList/Token < IsNull < unbounded Range
    // 5. Not(...)
    // 6. Or(...)
    let mut outcomes = HashMap::new();
    let all_true = MockOutcome::Exact {
        selected: vec![1],
        nulls: vec![],
    };
    let fts_query = SargableQuery::FullTextSearch(FullTextSearchQuery::new("hello".to_string()));
    let fts_key = fts_query.format("c_fts");
    let keys = [
        "c_eq_u_bt = 1",
        "c_eq_u_bm = 1",
        "c_eq_r_bt = 1",
        "c_eq_r_bm = 1",
        "c_in_2_bt IN [1,2]",
        "c_in_2_bm IN [1,2]",
        "c_in_4_bt IN [1,2,3,4]",
        "c_in_big IN [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]",
        "c_like LIKE 'ab%'",
        "c_rng2 >= 1 && c_rng2 <= 5",
        "c_rng1 >= 1",
        fts_key.as_str(),
        "c_null IS NULL",
        "true",
        "c_or1 = 1",
        "c_or2 = 2",
    ];
    for k in keys {
        outcomes.insert(k.to_string(), all_true.clone());
    }
    // `not_node` negates `c_not = 1`, so make `c_not = 1` match row 2 (leaving row 1 TRUE
    // after NOT so evaluation continues into `c_or1`).
    outcomes.insert(
        "c_not = 1".to_string(),
        MockOutcome::Exact {
            selected: vec![2],
            nulls: vec![],
        },
    );

    let eq_r_bt = q_search(
        "c_eq_r_bt",
        "BTree",
        SargableQuery::Equals(1_i64.into()),
        false,
    );
    let eq_r_bm = q_search(
        "c_eq_r_bm",
        "Bitmap",
        SargableQuery::Equals(1_i64.into()),
        false,
    );
    let in_big = q_search(
        "c_in_big",
        "BTree",
        SargableQuery::IsIn((0..17_i64).map(ScalarValue::from).collect()),
        false,
    );
    let not_node = ScalarIndexExpr::Not(Box::new(q_search(
        "c_not",
        "BTree",
        SargableQuery::Equals(1_i64.into()),
        false,
    )));
    let or_node = ScalarIndexExpr::Or(
        Box::new(q_search(
            "c_or1",
            "BTree",
            SargableQuery::Equals(1_i64.into()),
            false,
        )),
        Box::new(q_search(
            "c_or2",
            "BTree",
            SargableQuery::Equals(2_i64.into()),
            false,
        )),
    );

    let and_terms = vec![
        or_node,
        not_node,
        q_search(
            "c_unbounded",
            "BTree",
            SargableQuery::Range(Bound::Unbounded, Bound::Unbounded),
            false,
        ),
        q_search("c_null", "BTree", SargableQuery::IsNull(), false),
        q_search("c_fts", "Inverted", fts_query, true),
        q_search(
            "c_rng1",
            "BTree",
            SargableQuery::Range(Bound::Included(1_i64.into()), Bound::Unbounded),
            false,
        ),
        q_search(
            "c_rng2",
            "BTree",
            SargableQuery::Range(Bound::Included(1_i64.into()), Bound::Included(5_i64.into())),
            false,
        ),
        q_search(
            "c_like",
            "BTree",
            SargableQuery::LikePrefix("ab".into()),
            false,
        ),
        in_big,
        q_search(
            "c_in_4_bt",
            "BTree",
            SargableQuery::IsIn((1..=4_i64).map(ScalarValue::from).collect()),
            false,
        ),
        q_search(
            "c_in_2_bm",
            "Bitmap",
            SargableQuery::IsIn(vec![1_i64.into(), 2_i64.into()]),
            false,
        ),
        q_search(
            "c_in_2_bt",
            "BTree",
            SargableQuery::IsIn(vec![1_i64.into(), 2_i64.into()]),
            false,
        ),
        eq_r_bm.clone(),
        eq_r_bt.clone(),
        q_search(
            "c_eq_u_bm",
            "Bitmap",
            SargableQuery::Equals(1_i64.into()),
            false,
        ),
        q_search(
            "c_eq_u_bt",
            "BTree",
            SargableQuery::Equals(1_i64.into()),
            false,
        ),
    ];

    let big_and = and_terms
        .into_iter()
        .reduce(|a, b| ScalarIndexExpr::And(Box::new(a), Box::new(b)))
        .unwrap();
    // Force `eq_r_bt` and `eq_r_bm` to have occurrence_count = 2.
    let expr = ScalarIndexExpr::And(
        Box::new(big_and),
        Box::new(ScalarIndexExpr::Or(Box::new(eq_r_bt), Box::new(eq_r_bm))),
    );

    // Verify exact `conjunct_priority` ordering across all 16 tiers plus ready CSE (tier 0).
    let ready_leaf = q_search(
        "c_ready",
        "Bitmap",
        SargableQuery::Equals(1_i64.into()),
        false,
    );
    let ready_root = ScalarIndexExpr::Or(
        Box::new(ScalarIndexExpr::And(
            Box::new(expr.clone()),
            Box::new(ready_leaf.clone()),
        )),
        Box::new(ready_leaf.clone()),
    );
    let ready_ctx = EvalContext::new(&ready_root, false);
    let ScalarIndexExpr::Or(left_and, _) = &ready_root else {
        panic!("expected outer OR");
    };
    let ScalarIndexExpr::And(_, ready_in_and) = left_and.as_ref() else {
        panic!("expected ready leaf on RHS of AND");
    };
    let ScalarIndexExpr::Query(ready_search) = ready_in_and.as_ref() else {
        panic!("expected query leaf");
    };
    let ready_slot = ready_ctx.meta(ready_search, false).slot_id.unwrap();
    ready_ctx.slots[ready_slot]
        .native
        .set(Arc::new(NullableIndexExprResult::exact(
            NullableRowAddrMask::allow_nothing(),
        )))
        .unwrap();
    let mut sorted_conjuncts = Vec::new();
    collect_and_conjuncts(left_and, &mut sorted_conjuncts);
    sorted_conjuncts.sort_by_key(|c| conjunct_priority(c, &ready_ctx, false));
    let priority_labels: Vec<String> = sorted_conjuncts
        .iter()
        .map(|c| match c {
            ScalarIndexExpr::Query(s) => s.column.clone(),
            ScalarIndexExpr::Not(_) => "NOT".to_string(),
            ScalarIndexExpr::Or(lhs, _) => match lhs.as_ref() {
                ScalarIndexExpr::Query(s) => format!("OR({})", s.column),
                _ => "OR".to_string(),
            },
            ScalarIndexExpr::And(_, _) => "AND".to_string(),
        })
        .collect();
    assert_eq!(
        priority_labels,
        vec![
            "c_ready",
            "c_eq_u_bt",
            "c_eq_u_bm",
            "c_eq_r_bt",
            "c_eq_r_bm",
            "c_in_2_bt",
            "c_in_4_bt",
            "c_in_2_bm",
            "c_in_big",
            "c_like",
            "c_rng2",
            "c_rng1",
            "c_fts",
            "c_null",
            "c_unbounded",
            "NOT",
            "OR(c_or1)",
            "OR(c_eq_r_bt)",
        ]
    );

    let loader = MockLoader::new(outcomes);
    let res = expr.evaluate(&loader, &NoOpMetricsCollector).await.unwrap();
    let order = loader.stats.search_order.lock().unwrap().clone();
    // Initial selective probe runs alone and finishes first before bounded tail overlap.
    assert_eq!(order.first().map(String::as_str), Some("c_eq_u_bt = 1"));
    assert!(res.is_exact());
    assert_eq!(
        res.lower,
        RowAddrMask::from_allowed([1].into_iter().collect())
    );
}

#[tokio::test]
async fn test_exhaustive_kleene_and_interval_equivalence() {
    // Test leaves covering Exact/AtMost/AtLeast, empty/non-empty/universal, with NULLs
    // across complex nested AND / OR / NOT combinations against the recursive reference.
    let leaf_specs = [
        (
            "k_exact_some",
            MockOutcome::Exact {
                selected: vec![1, 2, 5],
                nulls: vec![5],
            },
            false,
        ),
        (
            "k_exact_null_only",
            MockOutcome::Exact {
                selected: vec![5],
                nulls: vec![5],
            },
            false,
        ),
        (
            "k_exact_empty",
            MockOutcome::Exact {
                selected: vec![],
                nulls: vec![],
            },
            false,
        ),
        (
            "k_exact_full_row_space",
            MockOutcome::Exact {
                selected: vec![0, 1, 2, 3, 4, 5, 6, 7],
                nulls: vec![],
            },
            false,
        ),
        (
            "k_at_most_some",
            MockOutcome::AtMost {
                selected: vec![1, 3, 5, 6],
                nulls: vec![5, 6],
            },
            true,
        ),
        (
            "k_at_most_empty",
            MockOutcome::AtMost {
                selected: vec![],
                nulls: vec![],
            },
            true,
        ),
        (
            "k_at_least_some",
            MockOutcome::AtLeast {
                selected: vec![2, 4, 5],
                nulls: vec![5],
            },
            true,
        ),
        (
            "k_at_least_empty",
            MockOutcome::AtLeast {
                selected: vec![],
                nulls: vec![],
            },
            true,
        ),
    ];

    let mut outcomes = HashMap::new();
    let mut exprs = Vec::new();
    for (i, (col, outcome, recheck)) in leaf_specs.iter().enumerate() {
        outcomes.insert(format!("{col} = {i}"), outcome.clone());
        exprs.push(q_search(
            col,
            if i % 2 == 0 { "BTree" } else { "Bitmap" },
            SargableQuery::Equals(ScalarValue::Int64(Some(i as i64))),
            *recheck,
        ));
    }
    // Also test BloomFilter, LabelList, and Token fallback hashing in CSE.
    outcomes.insert(
        "bf_col = 9".to_string(),
        MockOutcome::AtMost {
            selected: vec![1, 2, 5],
            nulls: vec![5],
        },
    );
    let bf = q_search(
        "bf_col",
        "BloomFilter",
        BloomFilterQuery::Equals(9_i64.into()),
        true,
    );
    let lbl_query = LabelListQuery::HasAnyLabel(vec![ScalarValue::Utf8(Some("a".into()))]);
    outcomes.insert(
        lbl_query.format("lbl"),
        MockOutcome::Exact {
            selected: vec![1, 4],
            nulls: vec![],
        },
    );
    let lbl = q_search("lbl", "LabelList", lbl_query, false);
    let tok_query = TokenQuery::TokensContains("tok".into());
    outcomes.insert(
        tok_query.format("txt"),
        MockOutcome::AtMost {
            selected: vec![2, 3],
            nulls: vec![],
        },
    );
    let tok = q_search("txt", "Inverted", tok_query, true);

    let loader = MockLoader::new(outcomes);

    // Build diverse trees including duplicates, nested NOTs, Refined intervals
    // (`AtLeast | Exact`, `AtLeast & AtMost`, `!AtMost`, etc.).
    let mut candidates = Vec::new();
    for a in &exprs {
        for b in &exprs {
            // a AND b, a OR b, NOT(a AND b), NOT(a) OR b, (a AND b) OR (a AND NOT(b))
            candidates.push(ScalarIndexExpr::And(
                Box::new(a.clone()),
                Box::new(b.clone()),
            ));
            candidates.push(ScalarIndexExpr::Or(
                Box::new(a.clone()),
                Box::new(b.clone()),
            ));
            candidates.push(ScalarIndexExpr::Not(Box::new(ScalarIndexExpr::And(
                Box::new(a.clone()),
                Box::new(b.clone()),
            ))));
            candidates.push(ScalarIndexExpr::Or(
                Box::new(ScalarIndexExpr::Not(Box::new(a.clone()))),
                Box::new(b.clone()),
            ));
            candidates.push(ScalarIndexExpr::Or(
                Box::new(ScalarIndexExpr::And(
                    Box::new(a.clone()),
                    Box::new(b.clone()),
                )),
                Box::new(ScalarIndexExpr::And(
                    Box::new(a.clone()),
                    Box::new(ScalarIndexExpr::Not(Box::new(b.clone()))),
                )),
            ));
        }
    }
    // Refined interval + non-SargableQuery CSE cases:
    let refined = ScalarIndexExpr::And(
        Box::new(ScalarIndexExpr::Or(
            Box::new(exprs[6].clone()), // AtLeast
            Box::new(exprs[0].clone()), // Exact
        )),
        Box::new(exprs[4].clone()), // AtMost -> Refined!
    );
    candidates.push(ScalarIndexExpr::Or(
        Box::new(ScalarIndexExpr::And(
            Box::new(refined.clone()),
            Box::new(bf.clone()),
        )),
        Box::new(ScalarIndexExpr::And(
            Box::new(ScalarIndexExpr::Not(Box::new(refined))),
            Box::new(ScalarIndexExpr::Or(
                Box::new(lbl.clone()),
                Box::new(tok.clone()),
            )),
        )),
    ));
    candidates.push(ScalarIndexExpr::And(
        Box::new(ScalarIndexExpr::Or(
            Box::new(bf.clone()),
            Box::new(lbl.clone()),
        )),
        Box::new(ScalarIndexExpr::Or(Box::new(bf), Box::new(tok))),
    ));

    for expr in candidates {
        // Check both `evaluate()` (`track_nulls=false`) and `evaluate_nullable()` (`track_nulls=true`).
        let actual = expr.evaluate(&loader, &NoOpMetricsCollector).await.unwrap();
        let expected = reference_eval_nullable(&expr, &loader, &NoOpMetricsCollector, false)
            .await
            .unwrap()
            .drop_nulls();
        assert_eq!(
            actual.lower, expected.lower,
            "evaluate lower mismatch on {expr}"
        );
        assert_eq!(
            actual.upper, expected.upper,
            "evaluate upper mismatch on {expr}"
        );
        assert_eq!(
            actual.is_exact(),
            expected.is_exact(),
            "evaluate is_exact mismatch on {expr}"
        );

        let actual_null = expr
            .evaluate_nullable(&loader, &NoOpMetricsCollector)
            .await
            .unwrap();
        let expected_null = reference_eval_nullable(&expr, &loader, &NoOpMetricsCollector, true)
            .await
            .unwrap();
        assert_eq!(
            actual_null.lower, expected_null.lower,
            "evaluate_nullable lower mismatch on {expr}"
        );
        assert_eq!(
            actual_null.upper, expected_null.upper,
            "evaluate_nullable upper mismatch on {expr}"
        );
        assert_eq!(
            actual_null.is_exact(),
            expected_null.is_exact(),
            "evaluate_nullable is_exact mismatch on {expr}"
        );
    }

    // Also test EvalValue directly across all (AllowList, BlockList) x (Exact, AtMost, AtLeast, Refined)
    // in Shared/Owned/ptr_eq combinations.
    let allow_lo = NullableRowAddrMask::AllowList(NullableRowAddrSet::new(
        RowAddrTreeMap::from_iter([1_u64, 5]),
        RowAddrTreeMap::from_iter([5_u64]),
    ));
    let allow_hi = NullableRowAddrMask::AllowList(NullableRowAddrSet::new(
        RowAddrTreeMap::from_iter([1_u64, 2, 5]),
        RowAddrTreeMap::from_iter([5_u64, 6]),
    ));
    let block_lo = NullableRowAddrMask::BlockList(NullableRowAddrSet::new(
        RowAddrTreeMap::from_iter([2_u64, 3, 5]),
        RowAddrTreeMap::from_iter([5_u64]),
    ));
    let block_hi = NullableRowAddrMask::BlockList(NullableRowAddrSet::new(
        RowAddrTreeMap::from_iter([3_u64, 5]),
        RowAddrTreeMap::from_iter([5_u64, 6]),
    ));
    let shapes = [
        NullableIndexExprResult::exact(allow_lo.clone()),
        NullableIndexExprResult::at_most(allow_hi.clone()),
        NullableIndexExprResult::at_least(allow_lo.clone()),
        NullableIndexExprResult::new(allow_lo, allow_hi),
        NullableIndexExprResult::exact(block_hi.clone()),
        NullableIndexExprResult::at_most(block_hi.clone()),
        NullableIndexExprResult::at_least(block_lo.clone()),
        NullableIndexExprResult::new(block_lo, block_hi),
    ];

    let mk_shared = |arc: &Arc<NullableIndexExprResult>| EvalValue::Shared {
        result: Arc::clone(arc),
        slot_id: None,
    };

    for a in &shapes {
        let shared_a = Arc::new(a.clone());
        // ptr_eq fast path
        let self_and = mk_shared(&shared_a)
            .bit_and(mk_shared(&shared_a))
            .into_owned();
        assert_eq!(self_and.lower, (a.clone() & a.clone()).lower);
        assert_eq!(self_and.upper, (a.clone() & a.clone()).upper);
        let self_or = mk_shared(&shared_a)
            .bit_or(mk_shared(&shared_a))
            .into_owned();
        assert_eq!(self_or.lower, (a.clone() | a.clone()).lower);
        assert_eq!(self_or.upper, (a.clone() | a.clone()).upper);

        for b in &shapes {
            let shared_b = Arc::new(b.clone());
            let ref_and = a.clone() & b.clone();
            let ref_or = a.clone() | b.clone();

            for (got_and, got_or) in [
                (
                    EvalValue::Owned(a.clone())
                        .bit_and(mk_shared(&shared_b))
                        .into_owned(),
                    EvalValue::Owned(a.clone())
                        .bit_or(mk_shared(&shared_b))
                        .into_owned(),
                ),
                (
                    mk_shared(&shared_a)
                        .bit_and(EvalValue::Owned(b.clone()))
                        .into_owned(),
                    mk_shared(&shared_a)
                        .bit_or(EvalValue::Owned(b.clone()))
                        .into_owned(),
                ),
                (
                    mk_shared(&shared_a)
                        .bit_and(mk_shared(&shared_b))
                        .into_owned(),
                    mk_shared(&shared_a)
                        .bit_or(mk_shared(&shared_b))
                        .into_owned(),
                ),
            ] {
                assert_eq!(got_and.lower, ref_and.lower);
                assert_eq!(got_and.upper, ref_and.upper);
                assert_eq!(got_and.is_exact(), ref_and.is_exact());
                assert_eq!(got_or.lower, ref_or.lower);
                assert_eq!(got_or.upper, ref_or.upper);
                assert_eq!(got_or.is_exact(), ref_or.is_exact());
            }
        }
    }
}

#[tokio::test]
async fn test_deep_2000_branch_left_deep_or_and_trees() {
    let mut outcomes = HashMap::new();
    for i in 0..2000_i64 {
        // For OR: branch 1999 matches row 1999.
        // For AND: all branches match row 42.
        outcomes.insert(
            format!("x = {i}"),
            MockOutcome::Exact {
                selected: if i == 1999 { vec![42, 1999] } else { vec![42] },
                nulls: vec![],
            },
        );
    }
    let loader = MockLoader::new(outcomes);

    // Build left-deep 2,000-branch OR tree.
    let mut deep_or = q_search("x", "BTree", SargableQuery::Equals(0_i64.into()), false);
    for i in 1..2000_i64 {
        deep_or = ScalarIndexExpr::Or(
            Box::new(deep_or),
            Box::new(q_search(
                "x",
                "BTree",
                SargableQuery::Equals(i.into()),
                false,
            )),
        );
    }
    // Evaluate unoptimized left-deep OR directly (must not stack overflow).
    let res_unopt_or = deep_or
        .evaluate(&loader, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert!(res_unopt_or.lower.selected(42));
    assert!(res_unopt_or.lower.selected(1999));

    // Optimize left-deep OR and evaluate.
    let opt_or = deep_or.optimize();
    let res_opt_or = opt_or
        .evaluate(&loader, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert_eq!(res_opt_or.lower, res_unopt_or.lower);

    // Build left-deep 2,000-branch AND tree.
    let mut deep_and = q_search("x", "BTree", SargableQuery::Equals(0_i64.into()), false);
    for i in 1..2000_i64 {
        deep_and = ScalarIndexExpr::And(
            Box::new(deep_and),
            Box::new(q_search(
                "x",
                "BTree",
                SargableQuery::Equals(i.into()),
                false,
            )),
        );
    }
    let res_unopt_and = deep_and
        .evaluate(&loader, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert_eq!(
        res_unopt_and.lower,
        RowAddrMask::from_allowed([42].into_iter().collect())
    );

    let opt_and = deep_and.optimize();
    let res_opt_and = opt_and
        .evaluate(&loader, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert_eq!(res_opt_and.lower, res_unopt_and.lower);
}

#[tokio::test]
async fn test_or_short_circuits_on_all_true_and_and_prefers_ready_cse() {
    // Case 1: OR short-circuits when a branch guarantees TRUE everywhere (`BlockList` empty).
    // `NOT(empty_query)` evaluates to `Exact(BlockList(empty, empty))` = `Exact(all_rows())`.
    let mut outcomes = HashMap::new();
    outcomes.insert(
        "a = 1".to_string(),
        MockOutcome::Exact {
            selected: vec![],
            nulls: vec![],
        },
    );
    outcomes.insert(
        "b = 2".to_string(),
        MockOutcome::AtMost {
            selected: vec![1, 2],
            nulls: vec![],
        },
    );

    let all_true_branch = ScalarIndexExpr::Not(Box::new(q_search(
        "a",
        "BTree",
        SargableQuery::Equals(1_i64.into()),
        false,
    )));
    let skipped_exact_branches: Vec<ScalarIndexExpr> = (0..20_i64)
        .map(|i| q_search("c", "BTree", SargableQuery::Equals(i.into()), false))
        .collect();
    let mut or_terms = vec![all_true_branch.clone()];
    or_terms.extend(skipped_exact_branches.clone());
    let or_exact = or_terms
        .into_iter()
        .reduce(|a, b| ScalarIndexExpr::Or(Box::new(a), Box::new(b)))
        .unwrap();

    let loader = MockLoader::new(outcomes.clone());
    let res_exact = or_exact
        .evaluate(&loader, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert!(res_exact.is_exact());
    assert_eq!(res_exact.lower, RowAddrMask::all_rows());
    // Branches 8..20 were never even spawned into FuturesUnordered.
    assert!(loader.stats.search_calls.load(Ordering::SeqCst) <= MAX_OR_CONCURRENCY);

    // When a skipped branch has `needs_recheck = true`, OR short-circuit must NOT claim exact.
    let mut or_recheck_terms = vec![all_true_branch];
    or_recheck_terms.extend(skipped_exact_branches);
    or_recheck_terms.push(q_search(
        "b",
        "ZoneMap",
        SargableQuery::Equals(2_i64.into()),
        true,
    ));
    let or_recheck = or_recheck_terms
        .into_iter()
        .reduce(|a, b| ScalarIndexExpr::Or(Box::new(a), Box::new(b)))
        .unwrap();

    let loader_recheck = MockLoader::new(outcomes);
    let res_recheck = or_recheck
        .evaluate(&loader_recheck, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert!(!res_recheck.is_exact());
    assert!(res_recheck.is_at_least());

    // Case 2: Tier 0 (`is_ready`) CSE leaf runs BEFORE unique point equality in `And`
    // and short-circuits if empty.
    let mut ready_outcomes = HashMap::new();
    ready_outcomes.insert(
        "ready_col = 1".to_string(),
        MockOutcome::Exact {
            selected: vec![],
            nulls: vec![],
        },
    );
    ready_outcomes.insert(
        "late_col = 2".to_string(),
        MockOutcome::Exact {
            selected: vec![1],
            nulls: vec![],
        },
    );
    let ready_leaf = q_search(
        "ready_col",
        "Bitmap",
        SargableQuery::Equals(1_i64.into()),
        false,
    );
    let unique_leaf = q_search(
        "late_col",
        "BTree",
        SargableQuery::Equals(2_i64.into()),
        false,
    );
    // Branches 0..8 populate `ready_leaf` (0..8 = MAX_OR_CONCURRENCY), then branch 8 runs
    // `And(unique_leaf, ready_leaf)` after `ready_leaf` is already initialized.
    let mut branches = Vec::new();
    for _ in 0..MAX_OR_CONCURRENCY {
        branches.push(ready_leaf.clone());
    }
    branches.push(ScalarIndexExpr::And(
        Box::new(unique_leaf),
        Box::new(ready_leaf),
    ));
    let expr_ready = branches
        .into_iter()
        .reduce(|a, b| ScalarIndexExpr::Or(Box::new(a), Box::new(b)))
        .unwrap();
    let loader_ready = MockLoader::new(ready_outcomes);
    let res_ready = expr_ready
        .evaluate(&loader_ready, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert_eq!(res_ready.lower, RowAddrMask::allow_nothing());
    // `ready_col = 1` resolved once in CSE; `late_col = 2` was skipped by Tier-0 short-circuit!
    let searched = loader_ready.stats.search_order.lock().unwrap().clone();
    assert_eq!(searched, vec!["ready_col = 1".to_string()]);
}

#[tokio::test]
async fn test_and_selective_probe_then_bounded_io_overlap() {
    // 1. Empty probe 0 short-circuits before any tail load starts.
    let mut empty_outcomes = HashMap::new();
    empty_outcomes.insert(
        "p0 = 1".to_string(),
        MockOutcome::Exact {
            selected: vec![],
            nulls: vec![],
        },
    );
    for i in 1..4 {
        empty_outcomes.insert(
            format!("p{i} = 1"),
            MockOutcome::Exact {
                selected: vec![1],
                nulls: vec![],
            },
        );
    }
    let empty_and = (0..4)
        .map(|i| {
            q_search(
                &format!("p{i}"),
                "BTree",
                SargableQuery::Equals(1_i64.into()),
                false,
            )
        })
        .reduce(|a, b| ScalarIndexExpr::And(Box::new(a), Box::new(b)))
        .unwrap();
    let empty_in_flight = Arc::new(AtomicUsize::new(0));
    let empty_max_in_flight = Arc::new(AtomicUsize::new(0));
    let mut empty_loader = MockLoader::new(empty_outcomes);
    {
        let in_flight = Arc::clone(&empty_in_flight);
        let max_in_flight = Arc::clone(&empty_max_in_flight);
        empty_loader.on_load = Some(Arc::new(move |_| {
            let in_flight = Arc::clone(&in_flight);
            let max_in_flight = Arc::clone(&max_in_flight);
            Box::pin(async move {
                let cur = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                max_in_flight.fetch_max(cur, Ordering::SeqCst);
                let _guard = InFlightGuard(in_flight);
                tokio::task::yield_now().await;
            })
        }));
    }
    let empty_res = tokio::time::timeout(
        Duration::from_secs(2),
        empty_and.evaluate(&empty_loader, &NoOpMetricsCollector),
    )
    .await
    .expect("empty probe timed out")
    .unwrap();
    assert!(empty_res.is_exact());
    assert_eq!(empty_res.lower, RowAddrMask::allow_nothing());
    assert_eq!(empty_loader.stats.load_calls.load(Ordering::SeqCst), 1);
    assert_eq!(empty_max_in_flight.load(Ordering::SeqCst), 1);

    // 2. 4 non-empty BTree Equals: probe 0 completes alone (`max_in_flight == 1`),
    // then tail 1..3 overlap concurrently via `Barrier::new(3)` (`max_in_flight == 3`).
    let mut four_outcomes = HashMap::new();
    four_outcomes.insert(
        "c0 = 1".to_string(),
        MockOutcome::Exact {
            selected: vec![1, 2],
            nulls: vec![],
        },
    );
    four_outcomes.insert(
        "c1 = 1".to_string(),
        MockOutcome::Exact {
            selected: vec![2, 3],
            nulls: vec![],
        },
    );
    four_outcomes.insert(
        "c2 = 1".to_string(),
        MockOutcome::Exact {
            selected: vec![2, 4],
            nulls: vec![],
        },
    );
    four_outcomes.insert(
        "c3 = 1".to_string(),
        MockOutcome::Exact {
            selected: vec![2, 5],
            nulls: vec![],
        },
    );
    let four_and = (0..4)
        .map(|i| {
            q_search(
                &format!("c{i}"),
                "BTree",
                SargableQuery::Equals(1_i64.into()),
                false,
            )
        })
        .reduce(|a, b| ScalarIndexExpr::And(Box::new(a), Box::new(b)))
        .unwrap();
    let four_in_flight = Arc::new(AtomicUsize::new(0));
    let four_max_in_flight = Arc::new(AtomicUsize::new(0));
    let probe_load_calls = Arc::new(AtomicUsize::new(0));
    let probe_max_in_flight = Arc::new(AtomicUsize::new(0));
    let tail_barrier = Arc::new(Barrier::new(3));
    let mut four_loader = MockLoader::new(four_outcomes);
    let stats_for_hook = Arc::clone(&four_loader.stats);
    {
        let in_flight = Arc::clone(&four_in_flight);
        let max_in_flight = Arc::clone(&four_max_in_flight);
        let probe_loads = Arc::clone(&probe_load_calls);
        let probe_max = Arc::clone(&probe_max_in_flight);
        let barrier = Arc::clone(&tail_barrier);
        four_loader.on_load = Some(Arc::new(move |col| {
            let col = col.to_string();
            let in_flight = Arc::clone(&in_flight);
            let max_in_flight = Arc::clone(&max_in_flight);
            let probe_loads = Arc::clone(&probe_loads);
            let probe_max = Arc::clone(&probe_max);
            let barrier = Arc::clone(&barrier);
            let stats = Arc::clone(&stats_for_hook);
            Box::pin(async move {
                let cur = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                max_in_flight.fetch_max(cur, Ordering::SeqCst);
                let _guard = InFlightGuard(in_flight);
                if col == "c0" {
                    tokio::task::yield_now().await;
                    probe_loads.store(stats.load_calls.load(Ordering::SeqCst), Ordering::SeqCst);
                    probe_max.store(max_in_flight.load(Ordering::SeqCst), Ordering::SeqCst);
                } else {
                    barrier.wait().await;
                }
            })
        }));
    }
    let four_res = tokio::time::timeout(
        Duration::from_secs(2),
        four_and.evaluate(&four_loader, &NoOpMetricsCollector),
    )
    .await
    .expect("4-conjunct overlap timed out")
    .unwrap();
    assert_eq!(probe_load_calls.load(Ordering::SeqCst), 1);
    assert_eq!(probe_max_in_flight.load(Ordering::SeqCst), 1);
    assert_eq!(four_max_in_flight.load(Ordering::SeqCst), 3);
    assert!(four_res.is_exact());
    assert_eq!(
        four_res.lower,
        RowAddrMask::from_allowed([2].into_iter().collect())
    );

    // 3. 20 non-empty BTree Equals at top level caps concurrency at MAX_AND_CONCURRENCY (8).
    let mut twenty_outcomes = HashMap::new();
    for i in 0..20_u64 {
        twenty_outcomes.insert(
            format!("t{i} = 1"),
            MockOutcome::Exact {
                selected: vec![7, 100 + i],
                nulls: vec![],
            },
        );
    }
    let twenty_and = (0..20)
        .map(|i| {
            q_search(
                &format!("t{i}"),
                "BTree",
                SargableQuery::Equals(1_i64.into()),
                false,
            )
        })
        .reduce(|a, b| ScalarIndexExpr::And(Box::new(a), Box::new(b)))
        .unwrap();
    let top_in_flight = Arc::new(AtomicUsize::new(0));
    let top_max_in_flight = Arc::new(AtomicUsize::new(0));
    let top_probe_max = Arc::new(AtomicUsize::new(0));
    let mut top_loader = MockLoader::new(twenty_outcomes);
    {
        let in_flight = Arc::clone(&top_in_flight);
        let max_in_flight = Arc::clone(&top_max_in_flight);
        let probe_max = Arc::clone(&top_probe_max);
        top_loader.on_load = Some(Arc::new(move |col| {
            let is_probe = col == "t0";
            let in_flight = Arc::clone(&in_flight);
            let max_in_flight = Arc::clone(&max_in_flight);
            let probe_max = Arc::clone(&probe_max);
            Box::pin(async move {
                let cur = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                max_in_flight.fetch_max(cur, Ordering::SeqCst);
                let _guard = InFlightGuard(in_flight);
                tokio::task::yield_now().await;
                if is_probe {
                    probe_max.store(max_in_flight.load(Ordering::SeqCst), Ordering::SeqCst);
                }
            })
        }));
    }
    let top_res = tokio::time::timeout(
        Duration::from_secs(2),
        twenty_and.evaluate(&top_loader, &NoOpMetricsCollector),
    )
    .await
    .expect("20-conjunct top-level AND timed out")
    .unwrap();
    assert_eq!(top_probe_max.load(Ordering::SeqCst), 1);
    assert_eq!(
        top_max_in_flight.load(Ordering::SeqCst),
        MAX_AND_CONCURRENCY
    );
    assert!(top_res.is_exact());
    assert_eq!(
        top_res.lower,
        RowAddrMask::from_allowed([7].into_iter().collect())
    );

    // 4. 20 non-empty BTree Equals nested under Or caps each branch at MAX_NESTED_AND_CONCURRENCY (4).
    let mut nested_outcomes = HashMap::new();
    for i in 0..20_u64 {
        nested_outcomes.insert(
            format!("a{i} = 1"),
            MockOutcome::Exact {
                selected: vec![11, 1000 + i],
                nulls: vec![],
            },
        );
        nested_outcomes.insert(
            format!("b{i} = 1"),
            MockOutcome::Exact {
                selected: vec![22, 2000 + i],
                nulls: vec![],
            },
        );
    }
    let branch_a = (0..20)
        .map(|i| {
            q_search(
                &format!("a{i}"),
                "BTree",
                SargableQuery::Equals(1_i64.into()),
                false,
            )
        })
        .reduce(|a, b| ScalarIndexExpr::And(Box::new(a), Box::new(b)))
        .unwrap();
    let branch_b = (0..20)
        .map(|i| {
            q_search(
                &format!("b{i}"),
                "BTree",
                SargableQuery::Equals(1_i64.into()),
                false,
            )
        })
        .reduce(|a, b| ScalarIndexExpr::And(Box::new(a), Box::new(b)))
        .unwrap();
    let nested_or = ScalarIndexExpr::Or(Box::new(branch_a), Box::new(branch_b));

    let a_in_flight = Arc::new(AtomicUsize::new(0));
    let a_max_in_flight = Arc::new(AtomicUsize::new(0));
    let b_in_flight = Arc::new(AtomicUsize::new(0));
    let b_max_in_flight = Arc::new(AtomicUsize::new(0));
    let mut nested_loader = MockLoader::new(nested_outcomes);
    {
        let a_cur = Arc::clone(&a_in_flight);
        let a_max = Arc::clone(&a_max_in_flight);
        let b_cur = Arc::clone(&b_in_flight);
        let b_max = Arc::clone(&b_max_in_flight);
        nested_loader.on_load = Some(Arc::new(move |col| {
            let (in_flight, max_in_flight) = if col.starts_with('a') {
                (Arc::clone(&a_cur), Arc::clone(&a_max))
            } else {
                (Arc::clone(&b_cur), Arc::clone(&b_max))
            };
            Box::pin(async move {
                let cur = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                max_in_flight.fetch_max(cur, Ordering::SeqCst);
                let _guard = InFlightGuard(in_flight);
                tokio::task::yield_now().await;
            })
        }));
    }
    let nested_res = tokio::time::timeout(
        Duration::from_secs(2),
        nested_or.evaluate(&nested_loader, &NoOpMetricsCollector),
    )
    .await
    .expect("nested Or(And) timed out")
    .unwrap();
    assert_eq!(
        a_max_in_flight.load(Ordering::SeqCst),
        MAX_NESTED_AND_CONCURRENCY
    );
    assert_eq!(
        b_max_in_flight.load(Ordering::SeqCst),
        MAX_NESTED_AND_CONCURRENCY
    );
    assert!(nested_res.is_exact());
    assert_eq!(
        nested_res.lower,
        RowAddrMask::from_allowed([11, 22].into_iter().collect())
    );
}

#[tokio::test]
async fn test_and_tail_overlap_short_circuits_in_flight_slow_conjuncts_preserving_recheck() {
    let mut outcomes = HashMap::new();
    outcomes.insert(
        "probe = 1".to_string(),
        MockOutcome::Exact {
            selected: vec![1],
            nulls: vec![],
        },
    );
    outcomes.insert(
        "fast_disjoint = 1".to_string(),
        MockOutcome::Exact {
            selected: vec![2],
            nulls: vec![],
        },
    );
    outcomes.insert(
        "slow_recheck = 1".to_string(),
        MockOutcome::AtMost {
            selected: vec![1],
            nulls: vec![],
        },
    );
    outcomes.insert(
        "slow_exact = 1".to_string(),
        MockOutcome::Exact {
            selected: vec![1],
            nulls: vec![],
        },
    );

    let expr = [
        q_search("probe", "BTree", SargableQuery::Equals(1_i64.into()), false),
        q_search(
            "fast_disjoint",
            "BTree",
            SargableQuery::Equals(1_i64.into()),
            false,
        ),
        q_search(
            "slow_recheck",
            "BTree",
            SargableQuery::Equals(1_i64.into()),
            true,
        ),
        q_search(
            "slow_exact",
            "BTree",
            SargableQuery::Equals(1_i64.into()),
            false,
        ),
    ]
    .into_iter()
    .reduce(|a, b| ScalarIndexExpr::And(Box::new(a), Box::new(b)))
    .unwrap();

    let started_slow = Arc::new(AtomicUsize::new(0));
    let cancelled_slow = Arc::new(AtomicUsize::new(0));
    let slow_ready = Arc::new(Notify::new());
    let mut loader = MockLoader::new(outcomes);
    {
        let started_slow = Arc::clone(&started_slow);
        let cancelled_slow = Arc::clone(&cancelled_slow);
        let slow_ready = Arc::clone(&slow_ready);
        loader.on_load = Some(Arc::new(move |col| {
            let col = col.to_string();
            let started_slow = Arc::clone(&started_slow);
            let cancelled_slow = Arc::clone(&cancelled_slow);
            let slow_ready = Arc::clone(&slow_ready);
            Box::pin(async move {
                match col.as_str() {
                    "slow_recheck" | "slow_exact" => {
                        let _guard = DropCounterGuard(cancelled_slow);
                        if started_slow.fetch_add(1, Ordering::SeqCst) + 1 == 2 {
                            slow_ready.notify_one();
                        }
                        std::future::pending::<()>().await;
                    }
                    "fast_disjoint" => {
                        let notified = slow_ready.notified();
                        if started_slow.load(Ordering::SeqCst) < 2 {
                            notified.await;
                        }
                    }
                    _ => {}
                }
            })
        }));
    }

    let res = tokio::time::timeout(
        Duration::from_secs(2),
        expr.evaluate(&loader, &NoOpMetricsCollector),
    )
    .await
    .expect("short-circuit test timed out")
    .unwrap();

    assert_eq!(started_slow.load(Ordering::SeqCst), 2);
    assert_eq!(cancelled_slow.load(Ordering::SeqCst), 2);
    assert!(!res.is_exact());
    assert!(res.is_at_most());
    assert_eq!(res.upper, RowAddrMask::allow_nothing());
    assert_eq!(loader.stats.search_calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn test_nested_leaf_search_budget_enforced_across_expression() {
    // Case 1: Nested CNF plan (`probe AND or_0 AND ... AND or_5`), where `probe` is a
    // selective leaf and each of the 6 `Or` conjuncts contains 8 unique leaves (48 tail
    // leaves; previously 5 `Or`s produced 40 simultaneous pending searches and 6 produced 48).
    // Verify deterministically:
    // - `probe` runs alone first (`probe_max_in_flight == 1`, `probe_search_calls == 1`).
    // - Independent tail searches overlap up to `MAX_LEAF_SEARCH_CONCURRENCY` (32) and never
    //   exceed 32 even while 48 leaf futures are simultaneously eligible.
    let mut cnf_outcomes = HashMap::new();
    cnf_outcomes.insert(
        "probe = 1".to_string(),
        MockOutcome::Exact {
            selected: vec![1, 2],
            nulls: vec![],
        },
    );
    for g in 0..6_usize {
        for b in 0..8_usize {
            cnf_outcomes.insert(
                format!("g{g}_b{b} = 1"),
                MockOutcome::Exact {
                    selected: if b == 0 {
                        vec![1]
                    } else {
                        vec![100 + b as u64]
                    },
                    nulls: vec![],
                },
            );
        }
    }

    let mut cnf_conjuncts = vec![q_search(
        "probe",
        "BTree",
        SargableQuery::Equals(1_i64.into()),
        false,
    )];
    for g in 0..6_usize {
        let disj = (0..8_usize)
            .map(|b| {
                q_search(
                    &format!("g{g}_b{b}"),
                    "BTree",
                    SargableQuery::Equals(1_i64.into()),
                    false,
                )
            })
            .reduce(|a, b| ScalarIndexExpr::Or(Box::new(a), Box::new(b)))
            .unwrap();
        cnf_conjuncts.push(disj);
    }
    let cnf_expr = cnf_conjuncts
        .into_iter()
        .reduce(|a, b| ScalarIndexExpr::And(Box::new(a), Box::new(b)))
        .unwrap()
        .optimize();

    let probe_max_in_flight = Arc::new(AtomicUsize::new(0));
    let probe_search_calls = Arc::new(AtomicUsize::new(0));
    let at_cap = Arc::new(Notify::new());
    let release_gate = Arc::new(Notify::new());
    let gate_open = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut cnf_loader = MockLoader::new(cnf_outcomes);
    let cnf_stats = Arc::clone(&cnf_loader.stats);
    {
        let probe_max = Arc::clone(&probe_max_in_flight);
        let probe_calls = Arc::clone(&probe_search_calls);
        let at_cap = Arc::clone(&at_cap);
        let release_gate = Arc::clone(&release_gate);
        let gate_open = Arc::clone(&gate_open);
        let stats = Arc::clone(&cnf_stats);
        cnf_loader.on_search = Some(Arc::new(move |key| {
            let is_probe = key == "probe = 1";
            let probe_max = Arc::clone(&probe_max);
            let probe_calls = Arc::clone(&probe_calls);
            let at_cap = Arc::clone(&at_cap);
            let release_gate = Arc::clone(&release_gate);
            let gate_open = Arc::clone(&gate_open);
            let stats = Arc::clone(&stats);
            Box::pin(async move {
                if is_probe {
                    tokio::task::yield_now().await;
                    probe_max.store(stats.max_in_flight.load(Ordering::SeqCst), Ordering::SeqCst);
                    probe_calls.store(stats.search_calls.load(Ordering::SeqCst), Ordering::SeqCst);
                    return;
                }
                let notified = release_gate.notified();
                if stats.in_flight.load(Ordering::SeqCst) == MAX_LEAF_SEARCH_CONCURRENCY {
                    at_cap.notify_one();
                }
                if !gate_open.load(Ordering::SeqCst) {
                    notified.await;
                }
            })
        }));
    }

    let eval_fut = cnf_expr.evaluate(&cnf_loader, &NoOpMetricsCollector);
    tokio::pin!(eval_fut);

    let cnf_res = tokio::time::timeout(Duration::from_secs(2), async {
        let at_cap_fut = at_cap.notified();
        tokio::select! {
            res = &mut eval_fut => panic!("evaluation finished before filling search budget: {res:?}"),
            _ = at_cap_fut => {}
        }
        // Poll repeatedly while all 32 leaf searches remain parked on `release_gate` so any
        // unthrottled 33rd..48th search would enter if the budget were per-node.
        for _ in 0..32 {
            let _ = futures::poll!(&mut eval_fut);
            tokio::task::yield_now().await;
        }
        assert_eq!(
            cnf_stats.in_flight.load(Ordering::SeqCst),
            MAX_LEAF_SEARCH_CONCURRENCY
        );
        assert_eq!(
            cnf_stats.max_in_flight.load(Ordering::SeqCst),
            MAX_LEAF_SEARCH_CONCURRENCY
        );
        gate_open.store(true, Ordering::SeqCst);
        release_gate.notify_waiters();
        eval_fut.await
    })
    .await
    .expect("nested CNF budget test timed out")
    .unwrap();

    assert_eq!(probe_max_in_flight.load(Ordering::SeqCst), 1);
    assert_eq!(probe_search_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        cnf_stats.max_in_flight.load(Ordering::SeqCst),
        MAX_LEAF_SEARCH_CONCURRENCY
    );
    assert_eq!(cnf_stats.search_calls.load(Ordering::SeqCst), 49);
    assert!(cnf_res.is_exact());
    assert_eq!(
        cnf_res.lower,
        RowAddrMask::from_allowed([1].into_iter().collect())
    );

    // Selective empty first probe on the same 48-leaf CNF plan must short-circuit before
    // starting any tail `Or` leaf search (`search_calls == 1`).
    let empty_cnf_loader = MockLoader::new(HashMap::new());
    let empty_cnf_res = cnf_expr
        .evaluate(&empty_cnf_loader, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert!(empty_cnf_res.is_exact());
    assert_eq!(empty_cnf_res.lower, RowAddrMask::allow_nothing());
    assert_eq!(
        empty_cnf_loader.stats.search_calls.load(Ordering::SeqCst),
        1
    );

    // Case 2: Multi-level nested plan (`Or` -> `And` -> `Not(Or)` / `Or`) with duplicate CSE
    // callers awaiting a parked `OnceCell` leaf (`shared = 1` repeated across 8 branches) plus
    // 39 other unique leaves across nested parents.
    // Parent nodes awaiting children and the 7 duplicate callers awaiting `shared = 1` must
    // consume ZERO slots so all 32 search slots still fill concurrently (`shared` + 31 unique
    // leaves = 32), while never exceeding `MAX_LEAF_SEARCH_CONCURRENCY` (32).
    let mut cse_outcomes = HashMap::new();
    cse_outcomes.insert(
        "shared = 1".to_string(),
        MockOutcome::Exact {
            selected: vec![5],
            nulls: vec![],
        },
    );
    for i in 0..39_usize {
        cse_outcomes.insert(
            format!("u{i} = 1"),
            MockOutcome::Exact {
                selected: vec![5, 1000 + i as u64],
                nulls: vec![],
            },
        );
    }

    let shared_leaf = q_search(
        "shared",
        "BTree",
        SargableQuery::Equals(1_i64.into()),
        false,
    );
    // 8 top-level `Or` branches:
    // - Branch 0: `shared` AND `(u0 OR ... OR u7)` AND `(u8 OR ... OR u15)`
    // - Branches 1..6 (6 branches): `shared` AND `(u_{start} OR ...)` (so all 6 probes hit
    //   duplicate `shared` `OnceCell` and park awaiting Branch 0's `shared` search!)
    // - Branch 7: `Not(Not(Or(shared, u16..u38)))` — nested parent nodes + duplicate `shared`
    //   + 23 unique leaves so >32 unique leaves are simultaneously eligible!
    let make_or_range = |range: std::ops::Range<usize>| {
        range
            .map(|i| {
                q_search(
                    &format!("u{i}"),
                    "BTree",
                    SargableQuery::Equals(1_i64.into()),
                    false,
                )
            })
            .reduce(|a, b| ScalarIndexExpr::Or(Box::new(a), Box::new(b)))
            .unwrap()
    };

    let mut top_branches = Vec::with_capacity(8);
    // Branches 0..5: each is `And(probe_u_i, Or(shared, 6 unique leaves))`.
    // Probes `u0..u5` finish immediately so all 6 branches enter Phase 3 concurrently,
    // starting `shared` once + 5 duplicate `shared` waiters + `6 * 6 = 36` unique leaves!
    for b in 0..6_usize {
        let probe_u = q_search(
            &format!("u{b}"),
            "BTree",
            SargableQuery::Equals(1_i64.into()),
            false,
        );
        let start = 6 + b * 5;
        let mut inner_disj = vec![shared_leaf.clone()];
        for i in start..start + 5 {
            inner_disj.push(q_search(
                &format!("u{i}"),
                "BTree",
                SargableQuery::Equals(1_i64.into()),
                false,
            ));
        }
        let inner_or = inner_disj
            .into_iter()
            .reduce(|a, b| ScalarIndexExpr::Or(Box::new(a), Box::new(b)))
            .unwrap();
        top_branches.push(ScalarIndexExpr::And(Box::new(probe_u), Box::new(inner_or)));
    }
    // Branches 6 and 7: `Not(Not(Or(shared, ...)))` adding nested `Not`/`Or` parents,
    // 2 more duplicate `shared` waiters (under `track_nulls = true` — a second CSE key with 1
    // leader + 1 duplicate waiter!), and `u36..u39`.
    let b6_or = ScalarIndexExpr::Or(
        Box::new(shared_leaf.clone()),
        Box::new(make_or_range(36..38)),
    );
    let b7_or = ScalarIndexExpr::Or(Box::new(shared_leaf), Box::new(make_or_range(38..40 - 1)));
    top_branches.push(ScalarIndexExpr::Not(Box::new(ScalarIndexExpr::Not(
        Box::new(b6_or),
    ))));
    top_branches.push(ScalarIndexExpr::Not(Box::new(ScalarIndexExpr::Not(
        Box::new(b7_or),
    ))));

    let nested_cse_expr = top_branches
        .into_iter()
        .reduce(|a, b| ScalarIndexExpr::Or(Box::new(a), Box::new(b)))
        .unwrap();

    let cse_at_cap = Arc::new(Notify::new());
    let cse_release = Arc::new(Notify::new());
    let cse_gate_open = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut cse_loader = MockLoader::new(cse_outcomes);
    let cse_stats = Arc::clone(&cse_loader.stats);
    {
        let at_cap = Arc::clone(&cse_at_cap);
        let release = Arc::clone(&cse_release);
        let gate_open = Arc::clone(&cse_gate_open);
        let stats = Arc::clone(&cse_stats);
        cse_loader.on_search = Some(Arc::new(move |key| {
            // Let the 6 selective probes (`u0..u5`) complete immediately so all 6 `And`
            // branches enter Phase 3 together and park duplicate `shared` callers + 33+ unique
            // leaf searches at the same time.
            let is_fast_probe = matches!(
                key,
                "u0 = 1" | "u1 = 1" | "u2 = 1" | "u3 = 1" | "u4 = 1" | "u5 = 1"
            );
            let at_cap = Arc::clone(&at_cap);
            let release = Arc::clone(&release);
            let gate_open = Arc::clone(&gate_open);
            let stats = Arc::clone(&stats);
            Box::pin(async move {
                if is_fast_probe {
                    return;
                }
                let notified = release.notified();
                if stats.in_flight.load(Ordering::SeqCst) == MAX_LEAF_SEARCH_CONCURRENCY {
                    at_cap.notify_one();
                }
                if !gate_open.load(Ordering::SeqCst) {
                    notified.await;
                }
            })
        }));
    }

    let cse_fut = nested_cse_expr.evaluate(&cse_loader, &NoOpMetricsCollector);
    tokio::pin!(cse_fut);

    let cse_res = tokio::time::timeout(Duration::from_secs(2), async {
        let at_cap_fut = cse_at_cap.notified();
        tokio::select! {
            res = &mut cse_fut => panic!("nested CSE evaluation finished before filling budget: {res:?}"),
            _ = at_cap_fut => {}
        }
        for _ in 0..32 {
            let _ = futures::poll!(&mut cse_fut);
            tokio::task::yield_now().await;
        }
        assert_eq!(
            cse_stats.in_flight.load(Ordering::SeqCst),
            MAX_LEAF_SEARCH_CONCURRENCY
        );
        assert_eq!(
            cse_stats.max_in_flight.load(Ordering::SeqCst),
            MAX_LEAF_SEARCH_CONCURRENCY
        );
        cse_gate_open.store(true, Ordering::SeqCst);
        cse_release.notify_waiters();
        cse_fut.await
    })
    .await
    .expect("nested CSE budget test timed out")
    .unwrap();

    assert_eq!(
        cse_stats.max_in_flight.load(Ordering::SeqCst),
        MAX_LEAF_SEARCH_CONCURRENCY
    );
    // 39 unique `u0..u38` leaves + 1 `shared` (track_nulls=false) + 1 `shared` (track_nulls=true) = 41
    // (whereas without CSE `shared` would run 8 times -> 47).
    assert_eq!(cse_stats.search_calls.load(Ordering::SeqCst), 41);
    assert!(cse_res.is_exact());
    assert!(cse_res.lower.selected(5));

    // Case 3: Cancelling an in-flight CSE `OnceCell` leader (via `And` short-circuit) while a
    // sibling `Or` branch awaits the same `OnceCell` and all 32 `search_semaphore` permits are
    // saturated must release the cancelled leader's permit and wake the parked waiter to finish.
    let mut cancel_outcomes = HashMap::new();
    for gate_col in [
        "probe_ok",
        "wait_leader",
        "gate0",
        "gate1",
        "gate2",
        "gate3",
        "gate4",
        "gate5",
    ] {
        cancel_outcomes.insert(
            format!("{gate_col} = 1"),
            MockOutcome::Exact {
                selected: vec![9],
                nulls: vec![],
            },
        );
    }
    cancel_outcomes.insert(
        "fast_false = 1".to_string(),
        MockOutcome::Exact {
            selected: vec![],
            nulls: vec![],
        },
    );
    cancel_outcomes.insert(
        "shared_cancel = 1".to_string(),
        MockOutcome::Exact {
            selected: vec![9],
            nulls: vec![],
        },
    );
    for i in 0..36_usize {
        cancel_outcomes.insert(
            format!("f{i} = 1"),
            MockOutcome::Exact {
                selected: vec![100 + i as u64],
                nulls: vec![],
            },
        );
    }

    let shared_cancel = q_search(
        "shared_cancel",
        "BTree",
        SargableQuery::Equals(1_i64.into()),
        false,
    );
    let branch0 = ScalarIndexExpr::And(
        Box::new(q_search(
            "probe_ok",
            "BTree",
            SargableQuery::Equals(1_i64.into()),
            false,
        )),
        Box::new(ScalarIndexExpr::And(
            Box::new(q_search(
                "fast_false",
                "BTree",
                SargableQuery::Equals(1_i64.into()),
                false,
            )),
            Box::new(shared_cancel.clone()),
        )),
    );
    // Gate Branch 1 and the 6 filler subtrees behind selective probes so Branch 0 always
    // enters Phase 3 and claims `shared_cancel` leader + `fast_false` first regardless of
    // `FuturesUnordered` LIFO poll order.
    let branch1 = ScalarIndexExpr::And(
        Box::new(q_search(
            "wait_leader",
            "BTree",
            SargableQuery::Equals(1_i64.into()),
            false,
        )),
        Box::new(shared_cancel),
    );
    let mut cancel_branches = vec![branch0, branch1];
    for g in 0..6_usize {
        let filler_or = (g * 6..(g + 1) * 6)
            .map(|i| {
                q_search(
                    &format!("f{i}"),
                    "BTree",
                    SargableQuery::Equals(1_i64.into()),
                    false,
                )
            })
            .reduce(|a, b| ScalarIndexExpr::Or(Box::new(a), Box::new(b)))
            .unwrap();
        let gated_filler = ScalarIndexExpr::And(
            Box::new(q_search(
                &format!("gate{g}"),
                "BTree",
                SargableQuery::Equals(1_i64.into()),
                false,
            )),
            Box::new(filler_or),
        );
        cancel_branches.push(gated_filler);
    }
    let cancel_expr = cancel_branches
        .into_iter()
        .reduce(|a, b| ScalarIndexExpr::Or(Box::new(a), Box::new(b)))
        .unwrap();

    let shared_attempts = Arc::new(AtomicUsize::new(0));
    let shared_cancelled = Arc::new(AtomicUsize::new(0));
    let branch0_ready_count = Arc::new(AtomicUsize::new(0));
    let branch0_ready = Arc::new(Notify::new());
    let saturated = Arc::new(Notify::new());
    let saturated_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let release_fillers = Arc::new(Notify::new());
    let fillers_open = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut cancel_loader = MockLoader::new(cancel_outcomes);
    let cancel_stats = Arc::clone(&cancel_loader.stats);
    {
        let shared_attempts = Arc::clone(&shared_attempts);
        let shared_cancelled = Arc::clone(&shared_cancelled);
        let branch0_ready_count = Arc::clone(&branch0_ready_count);
        let branch0_ready = Arc::clone(&branch0_ready);
        let saturated = Arc::clone(&saturated);
        let saturated_flag = Arc::clone(&saturated_flag);
        let release_fillers = Arc::clone(&release_fillers);
        let fillers_open = Arc::clone(&fillers_open);
        let stats = Arc::clone(&cancel_stats);
        cancel_loader.on_search = Some(Arc::new(move |key| {
            let key = key.to_string();
            let shared_attempts = Arc::clone(&shared_attempts);
            let shared_cancelled = Arc::clone(&shared_cancelled);
            let branch0_ready_count = Arc::clone(&branch0_ready_count);
            let branch0_ready = Arc::clone(&branch0_ready);
            let saturated = Arc::clone(&saturated);
            let saturated_flag = Arc::clone(&saturated_flag);
            let release_fillers = Arc::clone(&release_fillers);
            let fillers_open = Arc::clone(&fillers_open);
            let stats = Arc::clone(&stats);
            Box::pin(async move {
                if key == "probe_ok = 1" {
                    return;
                }
                if key == "wait_leader = 1" || key.starts_with("gate") {
                    let notified = branch0_ready.notified();
                    if branch0_ready_count.load(Ordering::SeqCst) < 2 {
                        notified.await;
                    }
                    return;
                }
                if stats.in_flight.load(Ordering::SeqCst) == MAX_LEAF_SEARCH_CONCURRENCY {
                    saturated_flag.store(true, Ordering::SeqCst);
                    saturated.notify_waiters();
                }
                match key.as_str() {
                    "shared_cancel = 1" => {
                        let attempt = shared_attempts.fetch_add(1, Ordering::SeqCst) + 1;
                        if attempt == 1 {
                            let _guard = DropCounterGuard(shared_cancelled);
                            if branch0_ready_count.fetch_add(1, Ordering::SeqCst) + 1 == 2 {
                                branch0_ready.notify_waiters();
                            }
                            std::future::pending::<()>().await;
                        }
                    }
                    "fast_false = 1" => {
                        if branch0_ready_count.fetch_add(1, Ordering::SeqCst) + 1 == 2 {
                            branch0_ready.notify_waiters();
                        }
                        let notified = saturated.notified();
                        if !saturated_flag.load(Ordering::SeqCst) {
                            notified.await;
                        }
                        tokio::task::yield_now().await;
                        fillers_open.store(true, Ordering::SeqCst);
                        release_fillers.notify_waiters();
                    }
                    _ => {
                        let notified = release_fillers.notified();
                        if !fillers_open.load(Ordering::SeqCst) {
                            notified.await;
                        }
                    }
                }
            })
        }));
    }

    let cancel_res = tokio::time::timeout(
        Duration::from_secs(2),
        cancel_expr.evaluate(&cancel_loader, &NoOpMetricsCollector),
    )
    .await
    .expect("CSE leader cancellation under saturated budget timed out")
    .unwrap();
    assert_eq!(shared_cancelled.load(Ordering::SeqCst), 1);
    assert_eq!(shared_attempts.load(Ordering::SeqCst), 2);
    assert_eq!(
        cancel_stats.max_in_flight.load(Ordering::SeqCst),
        MAX_LEAF_SEARCH_CONCURRENCY
    );
    assert!(cancel_res.lower.selected(9));
}

#[tokio::test]
async fn test_mixed_domain_cse_async_remap_does_not_deadlock() {
    let mut outcomes = HashMap::new();
    outcomes.insert(
        "addr0 = 1".to_string(),
        MockOutcome::Exact {
            selected: vec![1],
            nulls: vec![],
        },
    );
    outcomes.insert(
        "addr1 = 1".to_string(),
        MockOutcome::Exact {
            selected: vec![2, 3],
            nulls: vec![],
        },
    );
    outcomes.insert(
        "id_shared = 1".to_string(),
        MockOutcome::Exact {
            selected: vec![2],
            nulls: vec![],
        },
    );

    let id_shared = q_search(
        "id_shared",
        "BTree",
        SargableQuery::Equals(1_i64.into()),
        false,
    );
    let addr0 = q_search("addr0", "BTree", SargableQuery::Equals(1_i64.into()), false);
    let addr1 = q_search("addr1", "BTree", SargableQuery::Equals(1_i64.into()), false);

    // Order so `(addr1 AND id_shared)` enters `slot.row_addr.get_or_try_init` first inside
    // `eval_or`'s `in_flight` and yields during async remap, then `addr0` and `id_shared`
    // complete and `eval_or` attempts `reconcile_eval_domains` on `slot.row_addr`.
    let or_expr = ScalarIndexExpr::Or(
        Box::new(ScalarIndexExpr::And(
            Box::new(addr1.clone()),
            Box::new(id_shared.clone()),
        )),
        Box::new(ScalarIndexExpr::Or(
            Box::new(addr0.clone()),
            Box::new(id_shared.clone()),
        )),
    );

    let mut loader = MockLoader::new(outcomes.clone());
    loader.row_id_columns.insert("id_shared".to_string(), true);
    loader.yield_on_remap = true;
    loader.on_search = Some(Arc::new(|key| {
        let delay = key == "addr0 = 1";
        Box::pin(async move {
            if delay {
                tokio::task::yield_now().await;
            }
        })
    }));

    let res = tokio::time::timeout(
        Duration::from_millis(500),
        or_expr.evaluate(&loader, &NoOpMetricsCollector),
    )
    .await
    .expect("eval_or deadlocked on async CSE domain reconciliation")
    .unwrap();
    assert_eq!(loader.stats.remap_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        res.lower,
        RowAddrMask::from_allowed([1, 2].into_iter().collect())
    );

    // Same hazard in `eval_and` Phase 3: `addr0` (probe) -> Phase 3 overlaps
    // `(addr1 OR id_shared)` (acquires `slot.row_addr` inside `in_flight`) and `id_shared`.
    let and_expr = ScalarIndexExpr::And(
        Box::new(addr0),
        Box::new(ScalarIndexExpr::And(
            Box::new(ScalarIndexExpr::Or(
                Box::new(addr1),
                Box::new(id_shared.clone()),
            )),
            Box::new(id_shared),
        )),
    );
    let mut and_loader = MockLoader::new(outcomes);
    and_loader
        .row_id_columns
        .insert("id_shared".to_string(), true);
    and_loader.yield_on_remap = true;

    let and_res = tokio::time::timeout(
        Duration::from_millis(500),
        and_expr.evaluate(&and_loader, &NoOpMetricsCollector),
    )
    .await
    .expect("eval_and deadlocked on async CSE domain reconciliation")
    .unwrap();
    assert_eq!(and_loader.stats.remap_calls.load(Ordering::SeqCst), 1);
    assert_eq!(and_res.lower, RowAddrMask::allow_nothing());

    // Cross-branch ABBA: two sibling `Or` subtrees hold each other's `slot.row_addr`
    // leader inside their own `in_flight` while awaiting the opposite slot's conversion.
    let mut abba_outcomes = HashMap::new();
    abba_outcomes.insert(
        "addr_a = 1".to_string(),
        MockOutcome::Exact {
            selected: vec![1],
            nulls: vec![],
        },
    );
    abba_outcomes.insert(
        "addr_b = 1".to_string(),
        MockOutcome::Exact {
            selected: vec![2],
            nulls: vec![],
        },
    );
    abba_outcomes.insert(
        "id_a = 1".to_string(),
        MockOutcome::Exact {
            selected: vec![1],
            nulls: vec![],
        },
    );
    abba_outcomes.insert(
        "id_b = 1".to_string(),
        MockOutcome::Exact {
            selected: vec![2],
            nulls: vec![],
        },
    );
    let id_a = q_search("id_a", "BTree", SargableQuery::Equals(1_i64.into()), false);
    let id_b = q_search("id_b", "BTree", SargableQuery::Equals(1_i64.into()), false);
    let addr_a = q_search(
        "addr_a",
        "BTree",
        SargableQuery::Equals(1_i64.into()),
        false,
    );
    let addr_b = q_search(
        "addr_b",
        "BTree",
        SargableQuery::Equals(1_i64.into()),
        false,
    );
    // Subtree 1 (`Or`): `(addr_a AND id_a) OR addr_a OR id_b`
    // Subtree 2 (`Or`): `(addr_b AND id_b) OR addr_b OR id_a`
    let sub1 = ScalarIndexExpr::Or(
        Box::new(ScalarIndexExpr::And(
            Box::new(addr_a.clone()),
            Box::new(id_a.clone()),
        )),
        Box::new(ScalarIndexExpr::Or(
            Box::new(addr_a),
            Box::new(id_b.clone()),
        )),
    );
    let sub2 = ScalarIndexExpr::Or(
        Box::new(ScalarIndexExpr::And(
            Box::new(addr_b.clone()),
            Box::new(id_b),
        )),
        Box::new(ScalarIndexExpr::Or(Box::new(addr_b), Box::new(id_a))),
    );
    let abba_expr = ScalarIndexExpr::And(Box::new(sub1), Box::new(sub2));
    let mut abba_loader = MockLoader::new(abba_outcomes);
    abba_loader.row_id_columns.insert("id_a".to_string(), true);
    abba_loader.row_id_columns.insert("id_b".to_string(), true);
    abba_loader.yield_on_remap = true;

    let abba_res = tokio::time::timeout(
        Duration::from_millis(500),
        abba_expr.evaluate(&abba_loader, &NoOpMetricsCollector),
    )
    .await
    .expect("ABBA cross-branch async CSE domain reconciliation deadlocked")
    .unwrap();
    assert_eq!(abba_loader.stats.remap_calls.load(Ordering::SeqCst), 2);
    assert_eq!(
        abba_res.lower,
        RowAddrMask::from_allowed([1, 2].into_iter().collect())
    );
}
