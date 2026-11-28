// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Optimized evaluation engine for [`ScalarIndexExpr`].
//!
//! Evaluation runs in two phases:
//! 1. An iterative pre-evaluation CSE pass finds leaf searches that appear at
//!    least twice under the same `(column, index_name, index_type, needs_recheck,
//!    track_nulls, query)` key and assigns them shared `OnceCell` slots. Unique
//!    leaves (`count == 1`) bypass caching so their bitmaps drop right after
//!    folding.
//! 2. Contiguous `Or` regions evaluate branches via a local `FuturesUnordered`
//!    bounded at [`MAX_OR_CONCURRENCY`], folding and dropping each result
//!    immediately. Contiguous `And` regions sort conjuncts cheapest and most
//!    selective first and run in three phases: draining already-ready CSE
//!    leaves without I/O, probing the first remaining conjunct alone to
//!    short-circuit selective conjunctions before touching the tail, and
//!    overlapping remaining tail conjuncts via a local `FuturesUnordered`
//!    bounded at [`MAX_AND_CONCURRENCY`] (or [`MAX_NESTED_AND_CONCURRENCY`]
//!    when nested under an `Or`). Across the entire expression, simultaneous
//!    unique leaf index searches are capped at [`MAX_LEAF_SEARCH_CONCURRENCY`]
//!    at leaf-search start so parent nodes awaiting children and duplicate CSE
//!    callers awaiting a cached result consume no slots.

use std::any::TypeId;
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::ops::Bound;
use std::sync::Arc;

use async_recursion::async_recursion;
use datafusion_common::ScalarValue;
use futures::stream::{FuturesUnordered, StreamExt};
use lance_core::{Error, Result};
use lance_select::{
    MaskDomain, NullableIndexExprResult, NullableRowAddrMask, NullableRowAddrSet, RowSetOps,
};
use tokio::sync::{OnceCell, Semaphore};

use super::{
    AnyQuery, BloomFilterQuery, MetricsCollector, SargableQuery, SearchOptions, SearchResult,
    expression::{ScalarIndexExpr, ScalarIndexLoader, ScalarIndexSearch},
};

/// Maximum number of unique leaf index searches executing simultaneously across
/// a single [`ScalarIndexExpr`] evaluation.
///
/// Enforced via a per-evaluation semaphore acquired only when a unique leaf
/// search starts (`eval_search`): parent `Not` / `And` / `Or` nodes awaiting
/// children and duplicate CSE callers awaiting an in-flight `OnceCell` result
/// consume no slots, preventing cross-level deadlocks while strictly bounding
/// leaf fanout across arbitrarily nested plans.
const MAX_LEAF_SEARCH_CONCURRENCY: usize = 32;

/// Maximum number of branches polled concurrently within a single `Or` node so
/// completed branch masks fold and drop incrementally and `is_all_true` can
/// short-circuit before polling remaining branches.
const MAX_OR_CONCURRENCY: usize = 8;

/// Maximum number of tail conjuncts polled concurrently within a top-level
/// `And` node (outside any `Or`) after the initial selective probe.
const MAX_AND_CONCURRENCY: usize = 8;

/// Maximum number of tail conjuncts polled concurrently within an `And` node
/// nested under an `Or` after its initial selective probe; total simultaneous
/// leaf searches across the whole expression remain bounded by
/// [`MAX_LEAF_SEARCH_CONCURRENCY`].
const MAX_NESTED_AND_CONCURRENCY: usize = 4;

/// Maximum `IN (...)` list length treated as a narrow point-set lookup during
/// conjunctive ordering.
const NARROW_IS_IN_LIMIT: usize = 16;

/// Evaluate a [`ScalarIndexExpr`] using duplicate-leaf CSE, selective-probe +
/// bounded-overlap `And` evaluation, bounded `Or` concurrency, and borrowed-RHS
/// folding.
pub(super) async fn evaluate_expr(
    expr: &ScalarIndexExpr,
    index_loader: &dyn ScalarIndexLoader,
    metrics: &dyn MetricsCollector,
    track_nulls: bool,
) -> Result<NullableIndexExprResult> {
    let result = {
        let ctx = EvalContext::new(expr, track_nulls);
        eval_node(expr, &ctx, index_loader, metrics, track_nulls, false).await?
    };
    Ok(result.into_owned())
}

/// Compute a collision-resistant hash for `(index_type, needs_recheck, query)`
/// without string allocation for [`SargableQuery`] and [`BloomFilterQuery`].
pub(super) fn hash_search_key(search: &ScalarIndexSearch) -> u64 {
    let mut hasher = DefaultHasher::new();
    search.index_type.hash(&mut hasher);
    search.needs_recheck.hash(&mut hasher);
    let any_query = search.query.as_any();
    if let Some(sargable) = any_query.downcast_ref::<SargableQuery>() {
        TypeId::of::<SargableQuery>().hash(&mut hasher);
        hash_sargable_query(sargable, &search.column, &mut hasher);
    } else if let Some(bloom) = any_query.downcast_ref::<BloomFilterQuery>() {
        TypeId::of::<BloomFilterQuery>().hash(&mut hasher);
        hash_bloom_query(bloom, &mut hasher);
    } else {
        any_query.type_id().hash(&mut hasher);
        search.query.format(&search.column).hash(&mut hasher);
    }
    hasher.finish()
}

/// Exact structural equality for leaf searches in evaluation CSE (excluding
/// `fragment_bitmap`, which does not affect leaf evaluation).
pub(super) fn canonical_search_eq(a: &ScalarIndexSearch, b: &ScalarIndexSearch) -> bool {
    a.column == b.column
        && a.index_name == b.index_name
        && a.index_type == b.index_type
        && a.needs_recheck == b.needs_recheck
        && a.query.as_ref() == b.query.as_ref()
}

fn hash_bound(bound: &Bound<ScalarValue>, hasher: &mut impl Hasher) {
    match bound {
        Bound::Unbounded => 0u8.hash(hasher),
        Bound::Included(val) => {
            1u8.hash(hasher);
            val.hash(hasher);
        }
        Bound::Excluded(val) => {
            2u8.hash(hasher);
            val.hash(hasher);
        }
    }
}

fn hash_sargable_query(query: &SargableQuery, column: &str, hasher: &mut impl Hasher) {
    std::mem::discriminant(query).hash(hasher);
    match query {
        SargableQuery::Range(lower, upper) => {
            hash_bound(lower, hasher);
            hash_bound(upper, hasher);
        }
        SargableQuery::IsIn(values) => {
            values.len().hash(hasher);
            for val in values {
                val.hash(hasher);
            }
        }
        SargableQuery::Equals(val) | SargableQuery::LikePrefix(val) => {
            val.hash(hasher);
        }
        SargableQuery::IsNull() => {}
        SargableQuery::FullTextSearch(fts) => {
            query.format(column).hash(hasher);
            fts.limit.hash(hasher);
            fts.wand_factor.map(f32::to_bits).hash(hasher);
        }
    }
}

fn hash_bloom_query(query: &BloomFilterQuery, hasher: &mut impl Hasher) {
    std::mem::discriminant(query).hash(hasher);
    match query {
        BloomFilterQuery::Equals(val) => val.hash(hasher),
        BloomFilterQuery::IsNull() => {}
        BloomFilterQuery::IsIn(values) => {
            values.len().hash(hasher);
            for val in values {
                val.hash(hasher);
            }
        }
    }
}

struct CanonicalLeafEntry<'a> {
    search: &'a ScalarIndexSearch,
    track_nulls: bool,
    ptrs: Vec<usize>,
}

#[derive(Clone, Copy)]
struct LeafMeta {
    slot_id: Option<usize>,
    occurrence_count: usize,
}

struct CseSlot {
    native: OnceCell<Arc<NullableIndexExprResult>>,
    row_addr: OnceCell<Arc<NullableIndexExprResult>>,
}

impl CseSlot {
    fn new() -> Self {
        Self {
            native: OnceCell::new(),
            row_addr: OnceCell::new(),
        }
    }
}

struct EvalContext {
    slots: Vec<CseSlot>,
    leaf_meta: HashMap<(usize, bool), LeafMeta>,
    search_semaphore: Semaphore,
}

impl EvalContext {
    fn new(expr: &ScalarIndexExpr, track_nulls: bool) -> Self {
        let mut stack = vec![(expr, track_nulls)];
        let mut buckets: HashMap<(&str, &str, bool, u64), Vec<CanonicalLeafEntry<'_>>> =
            HashMap::new();

        while let Some((node, cur_track_nulls)) = stack.pop() {
            match node {
                ScalarIndexExpr::Not(inner) => {
                    stack.push((inner.as_ref(), true));
                }
                ScalarIndexExpr::And(lhs, rhs) | ScalarIndexExpr::Or(lhs, rhs) => {
                    stack.push((rhs.as_ref(), cur_track_nulls));
                    stack.push((lhs.as_ref(), cur_track_nulls));
                }
                ScalarIndexExpr::Query(search) => {
                    let ptr = search as *const ScalarIndexSearch as usize;
                    let qhash = hash_search_key(search);
                    let bucket_key = (
                        search.column.as_str(),
                        search.index_name.as_str(),
                        cur_track_nulls,
                        qhash,
                    );
                    let entries = buckets.entry(bucket_key).or_default();
                    if let Some(entry) = entries.iter_mut().find(|entry| {
                        entry.track_nulls == cur_track_nulls
                            && canonical_search_eq(entry.search, search)
                    }) {
                        entry.ptrs.push(ptr);
                    } else {
                        entries.push(CanonicalLeafEntry {
                            search,
                            track_nulls: cur_track_nulls,
                            ptrs: vec![ptr],
                        });
                    }
                }
            }
        }

        let mut slots = Vec::new();
        let mut leaf_meta = HashMap::new();
        for entries in buckets.into_values() {
            for entry in entries {
                let occurrence_count = entry.ptrs.len();
                let slot_id = if occurrence_count >= 2 {
                    let id = slots.len();
                    slots.push(CseSlot::new());
                    Some(id)
                } else {
                    None
                };
                for ptr in entry.ptrs {
                    leaf_meta.insert(
                        (ptr, entry.track_nulls),
                        LeafMeta {
                            slot_id,
                            occurrence_count,
                        },
                    );
                }
            }
        }

        Self {
            slots,
            leaf_meta,
            search_semaphore: Semaphore::new(MAX_LEAF_SEARCH_CONCURRENCY),
        }
    }

    fn meta(&self, search: &ScalarIndexSearch, track_nulls: bool) -> LeafMeta {
        let ptr = search as *const ScalarIndexSearch as usize;
        self.leaf_meta
            .get(&(ptr, track_nulls))
            .copied()
            .unwrap_or(LeafMeta {
                slot_id: None,
                occurrence_count: 1,
            })
    }

    fn is_ready(&self, search: &ScalarIndexSearch, track_nulls: bool) -> bool {
        let Some(slot_id) = self.meta(search, track_nulls).slot_id else {
            return false;
        };
        self.slots
            .get(slot_id)
            .is_some_and(|slot| slot.native.get().is_some())
    }
}

/// Evaluation value that distinguishes unique owned masks from shared CSE masks.
enum EvalValue {
    Owned(NullableIndexExprResult),
    Shared {
        result: Arc<NullableIndexExprResult>,
        slot_id: Option<usize>,
    },
}

impl EvalValue {
    fn as_result(&self) -> &NullableIndexExprResult {
        match self {
            Self::Owned(res) => res,
            Self::Shared { result, .. } => result.as_ref(),
        }
    }

    fn into_owned(self) -> NullableIndexExprResult {
        match self {
            Self::Owned(res) => res,
            Self::Shared { result, .. } => Arc::try_unwrap(result).unwrap_or_else(|a| (*a).clone()),
        }
    }

    fn bit_and(self, rhs: Self) -> Self {
        Self::combine(self, rhs, |a, b| a & b, |a, b| a & b)
    }

    fn bit_or(self, rhs: Self) -> Self {
        Self::combine(self, rhs, |a, b| a | b, |a, b| a | b)
    }

    fn combine(
        lhs: Self,
        rhs: Self,
        owned_op: fn(NullableIndexExprResult, NullableIndexExprResult) -> NullableIndexExprResult,
        borrowed_op: fn(
            NullableIndexExprResult,
            &NullableIndexExprResult,
        ) -> NullableIndexExprResult,
    ) -> Self {
        match (lhs, rhs) {
            (Self::Owned(a), Self::Owned(b)) => Self::Owned(owned_op(a, b)),
            (Self::Owned(a), Self::Shared { result: b, .. })
            | (Self::Shared { result: b, .. }, Self::Owned(a)) => {
                Self::Owned(borrowed_op(a, b.as_ref()))
            }
            (
                Self::Shared {
                    result: a,
                    slot_id: slot_a,
                },
                Self::Shared { result: b, .. },
            ) => {
                if Arc::ptr_eq(&a, &b) {
                    return Self::Shared {
                        result: a,
                        slot_id: slot_a,
                    };
                }
                if result_size_hint(a.as_ref()) <= result_size_hint(b.as_ref()) {
                    Self::Owned(borrowed_op((*a).clone(), b.as_ref()))
                } else {
                    Self::Owned(borrowed_op((*b).clone(), a.as_ref()))
                }
            }
        }
    }
}

impl std::ops::Not for EvalValue {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self::Owned(!self.into_owned())
    }
}

fn mask_size_hint(mask: &NullableRowAddrMask) -> u64 {
    match mask {
        NullableRowAddrMask::AllowList(set) | NullableRowAddrMask::BlockList(set) => set
            .selected_rows()
            .len()
            .unwrap_or(u64::MAX / 4)
            .saturating_add(set.null_rows().len().unwrap_or(u64::MAX / 4)),
    }
}

fn result_size_hint(res: &NullableIndexExprResult) -> u64 {
    if res.is_exact() {
        mask_size_hint(&res.lower)
    } else {
        mask_size_hint(&res.lower).saturating_add(mask_size_hint(&res.upper))
    }
}

fn search_result_to_nullable(result: SearchResult, domain: MaskDomain) -> NullableIndexExprResult {
    let result = match result {
        SearchResult::Exact(mask) => {
            NullableIndexExprResult::exact(NullableRowAddrMask::AllowList(mask))
        }
        SearchResult::AtMost(mask) => {
            NullableIndexExprResult::at_most(NullableRowAddrMask::AllowList(mask))
        }
        SearchResult::AtLeast(mask) => {
            NullableIndexExprResult::at_least(NullableRowAddrMask::AllowList(mask))
        }
    };
    result.with_domain(domain)
}

async fn into_row_address(
    val: EvalValue,
    ctx: &EvalContext,
    index_loader: &dyn ScalarIndexLoader,
) -> Result<EvalValue> {
    if val.as_result().domain == MaskDomain::RowAddress {
        return Ok(val);
    }
    match val {
        EvalValue::Owned(res) => Ok(EvalValue::Owned(
            index_loader.row_id_result_to_row_addrs(res).await?,
        )),
        EvalValue::Shared { result, slot_id } => {
            if let Some(id) = slot_id
                && let Some(slot) = ctx.slots.get(id)
            {
                let arc = slot
                    .row_addr
                    .get_or_try_init(|| async {
                        let converted = index_loader
                            .row_id_result_to_row_addrs((*result).clone())
                            .await?;
                        Ok::<_, Error>(Arc::new(converted))
                    })
                    .await?;
                return Ok(EvalValue::Shared {
                    result: Arc::clone(arc),
                    slot_id: None,
                });
            }
            let owned = Arc::try_unwrap(result).unwrap_or_else(|a| (*a).clone());
            Ok(EvalValue::Owned(
                index_loader.row_id_result_to_row_addrs(owned).await?,
            ))
        }
    }
}

/// Reconcile mismatched mask domains before combining two [`EvalValue`]s with
/// `And`/`Or`, converting whichever side is row-id-domain into row-address
/// domain and memoizing conversions on shared CSE slots.
async fn reconcile_eval_domains(
    lhs: EvalValue,
    rhs: EvalValue,
    ctx: &EvalContext,
    index_loader: &dyn ScalarIndexLoader,
) -> Result<(EvalValue, EvalValue)> {
    match (lhs.as_result().domain, rhs.as_result().domain) {
        (MaskDomain::RowId, MaskDomain::RowAddress) => {
            Ok((into_row_address(lhs, ctx, index_loader).await?, rhs))
        }
        (MaskDomain::RowAddress, MaskDomain::RowId) => {
            Ok((lhs, into_row_address(rhs, ctx, index_loader).await?))
        }
        _ => Ok((lhs, rhs)),
    }
}

#[async_recursion]
async fn eval_node(
    expr: &ScalarIndexExpr,
    ctx: &EvalContext,
    index_loader: &dyn ScalarIndexLoader,
    metrics: &dyn MetricsCollector,
    track_nulls: bool,
    in_disjunction: bool,
) -> Result<EvalValue> {
    match expr {
        ScalarIndexExpr::Not(inner) => {
            // NOT needs the child's NULL rows to preserve SQL three-valued
            // logic. Once enabled, keep tracking through the whole subtree.
            let val = eval_node(
                inner.as_ref(),
                ctx,
                index_loader,
                metrics,
                true,
                in_disjunction,
            )
            .await?;
            Ok(!val)
        }
        ScalarIndexExpr::And(_, _) => {
            eval_and(
                expr,
                ctx,
                index_loader,
                metrics,
                track_nulls,
                in_disjunction,
            )
            .await
        }
        ScalarIndexExpr::Or(_, _) => eval_or(expr, ctx, index_loader, metrics, track_nulls).await,
        ScalarIndexExpr::Query(search) => {
            eval_query(search, ctx, index_loader, metrics, track_nulls).await
        }
    }
}

async fn eval_query(
    search: &ScalarIndexSearch,
    ctx: &EvalContext,
    index_loader: &dyn ScalarIndexLoader,
    metrics: &dyn MetricsCollector,
    track_nulls: bool,
) -> Result<EvalValue> {
    if let Some(slot_id) = ctx.meta(search, track_nulls).slot_id {
        let slot = ctx.slots.get(slot_id).ok_or_else(|| {
            Error::internal(format!(
                "CSE slot {slot_id} missing while evaluating index search on '{}'",
                search.column
            ))
        })?;
        let arc = slot
            .native
            .get_or_try_init(|| async {
                let _permit = ctx.search_semaphore.acquire().await.map_err(|_| {
                    Error::internal(format!(
                        "leaf search semaphore closed while evaluating '{}'",
                        search.column
                    ))
                })?;
                let res = eval_search(search, index_loader, metrics, track_nulls).await?;
                Ok::<_, Error>(Arc::new(res))
            })
            .await?;
        Ok(EvalValue::Shared {
            result: Arc::clone(arc),
            slot_id: Some(slot_id),
        })
    } else {
        let _permit = ctx.search_semaphore.acquire().await.map_err(|_| {
            Error::internal(format!(
                "leaf search semaphore closed while evaluating '{}'",
                search.column
            ))
        })?;
        let res = eval_search(search, index_loader, metrics, track_nulls).await?;
        Ok(EvalValue::Owned(res))
    }
}

async fn eval_search(
    search: &ScalarIndexSearch,
    index_loader: &dyn ScalarIndexLoader,
    metrics: &dyn MetricsCollector,
    track_nulls: bool,
) -> Result<NullableIndexExprResult> {
    let index = index_loader
        .load_index(&search.column, &search.index_name, metrics)
        .await?;
    let search_result = index
        .search_with_options(
            search.query.as_ref(),
            SearchOptions::default().with_track_nulls(track_nulls),
            metrics,
        )
        .await?;
    // Leave address-domain results as-is and tag row-id-domain results
    // accordingly; domain reconciliation happens when combining mismatched
    // siblings in `eval_and` / `eval_or`.
    let domain = if index.results_are_row_addresses() {
        MaskDomain::RowAddress
    } else {
        MaskDomain::RowId
    };
    Ok(search_result_to_nullable(search_result, domain))
}

fn collect_or_branches<'a>(expr: &'a ScalarIndexExpr, out: &mut Vec<&'a ScalarIndexExpr>) {
    let mut stack = vec![expr];
    while let Some(node) = stack.pop() {
        match node {
            ScalarIndexExpr::Or(lhs, rhs) => {
                stack.push(rhs.as_ref());
                stack.push(lhs.as_ref());
            }
            other => out.push(other),
        }
    }
}

fn collect_and_conjuncts<'a>(expr: &'a ScalarIndexExpr, out: &mut Vec<&'a ScalarIndexExpr>) {
    let mut stack = vec![expr];
    while let Some(node) = stack.pop() {
        match node {
            ScalarIndexExpr::And(lhs, rhs) => {
                stack.push(rhs.as_ref());
                stack.push(lhs.as_ref());
            }
            other => out.push(other),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct ConjunctPriority {
    tier: u8,
    sub_tier: u8,
    flag_a: bool,
    list_len: usize,
    flag_b: bool,
}

fn is_bitmap_index(index_type: &str) -> bool {
    index_type.eq_ignore_ascii_case("Bitmap")
}

fn conjunct_priority(
    expr: &ScalarIndexExpr,
    ctx: &EvalContext,
    track_nulls: bool,
) -> ConjunctPriority {
    match expr {
        ScalarIndexExpr::Query(search) => {
            if ctx.is_ready(search, track_nulls) {
                return ConjunctPriority {
                    tier: 0,
                    sub_tier: 0,
                    flag_a: false,
                    list_len: 0,
                    flag_b: false,
                };
            }

            let is_repeated = ctx.meta(search, track_nulls).occurrence_count > 1;
            let is_bitmap = is_bitmap_index(&search.index_type);
            let any_query = search.query.as_any();

            if let Some(sargable) = any_query.downcast_ref::<SargableQuery>() {
                match sargable {
                    SargableQuery::Equals(_) => ConjunctPriority {
                        tier: 1,
                        sub_tier: 0,
                        flag_a: is_repeated,
                        list_len: 0,
                        flag_b: is_bitmap,
                    },
                    SargableQuery::IsIn(values) if values.len() <= NARROW_IS_IN_LIMIT => {
                        ConjunctPriority {
                            tier: 2,
                            sub_tier: 0,
                            flag_a: is_bitmap,
                            list_len: values.len(),
                            flag_b: is_repeated,
                        }
                    }
                    SargableQuery::IsIn(values) => ConjunctPriority {
                        tier: 3,
                        sub_tier: 0,
                        flag_a: is_bitmap,
                        list_len: values.len(),
                        flag_b: is_repeated,
                    },
                    SargableQuery::LikePrefix(_) => ConjunctPriority {
                        tier: 3,
                        sub_tier: 1,
                        flag_a: is_repeated,
                        list_len: 0,
                        flag_b: is_bitmap,
                    },
                    SargableQuery::Range(Bound::Unbounded, Bound::Unbounded) => ConjunctPriority {
                        tier: 3,
                        sub_tier: 6,
                        flag_a: false,
                        list_len: 0,
                        flag_b: false,
                    },
                    SargableQuery::Range(Bound::Unbounded, _)
                    | SargableQuery::Range(_, Bound::Unbounded) => ConjunctPriority {
                        tier: 3,
                        sub_tier: 3,
                        flag_a: is_repeated,
                        list_len: 0,
                        flag_b: is_bitmap,
                    },
                    SargableQuery::Range(_, _) => ConjunctPriority {
                        tier: 3,
                        sub_tier: 2,
                        flag_a: is_repeated,
                        list_len: 0,
                        flag_b: is_bitmap,
                    },
                    SargableQuery::FullTextSearch(_) => ConjunctPriority {
                        tier: 3,
                        sub_tier: 4,
                        flag_a: false,
                        list_len: 0,
                        flag_b: false,
                    },
                    SargableQuery::IsNull() => ConjunctPriority {
                        tier: 3,
                        sub_tier: 5,
                        flag_a: false,
                        list_len: 0,
                        flag_b: false,
                    },
                }
            } else if let Some(bloom) = any_query.downcast_ref::<BloomFilterQuery>() {
                match bloom {
                    BloomFilterQuery::Equals(_) => ConjunctPriority {
                        tier: 1,
                        sub_tier: 0,
                        flag_a: is_repeated,
                        list_len: 0,
                        flag_b: is_bitmap,
                    },
                    BloomFilterQuery::IsIn(values) if values.len() <= NARROW_IS_IN_LIMIT => {
                        ConjunctPriority {
                            tier: 2,
                            sub_tier: 0,
                            flag_a: is_bitmap,
                            list_len: values.len(),
                            flag_b: is_repeated,
                        }
                    }
                    BloomFilterQuery::IsIn(values) => ConjunctPriority {
                        tier: 3,
                        sub_tier: 0,
                        flag_a: is_bitmap,
                        list_len: values.len(),
                        flag_b: is_repeated,
                    },
                    BloomFilterQuery::IsNull() => ConjunctPriority {
                        tier: 3,
                        sub_tier: 5,
                        flag_a: false,
                        list_len: 0,
                        flag_b: false,
                    },
                }
            } else {
                ConjunctPriority {
                    tier: 3,
                    sub_tier: 4,
                    flag_a: false,
                    list_len: 0,
                    flag_b: false,
                }
            }
        }
        ScalarIndexExpr::Not(_) => ConjunctPriority {
            tier: 4,
            sub_tier: 0,
            flag_a: false,
            list_len: 0,
            flag_b: false,
        },
        ScalarIndexExpr::Or(_, _) => ConjunctPriority {
            tier: 5,
            sub_tier: 0,
            flag_a: false,
            list_len: 0,
            flag_b: false,
        },
        ScalarIndexExpr::And(_, _) => ConjunctPriority {
            tier: 5,
            sub_tier: 1,
            flag_a: false,
            list_len: 0,
            flag_b: false,
        },
    }
}

fn is_all_true(result: &NullableIndexExprResult) -> bool {
    let is_unblocked_empty = |mask: &NullableRowAddrMask| {
        matches!(
            mask,
            NullableRowAddrMask::BlockList(set)
                if set.selected_rows().is_empty() && set.null_rows().is_empty()
        )
    };
    is_unblocked_empty(&result.lower) && is_unblocked_empty(&result.upper)
}

fn has_no_true_rows(set: &NullableRowAddrSet) -> bool {
    if set.selected_rows().is_empty() {
        true
    } else if set.null_rows().is_empty() {
        false
    } else {
        set.true_rows().is_empty()
    }
}

fn is_strictly_false_everywhere(mask: &NullableRowAddrMask) -> bool {
    matches!(
        mask,
        NullableRowAddrMask::AllowList(set)
            if set.selected_rows().is_empty() && set.null_rows().is_empty()
    )
}

fn can_short_circuit_and(result: &NullableIndexExprResult, track_nulls: bool) -> bool {
    if track_nulls {
        // Under NOT (or evaluate_nullable), NULL rows must be preserved because
        // NOT(NULL) == NULL != TRUE and NULL & FALSE == FALSE.
        is_strictly_false_everywhere(&result.upper) && is_strictly_false_everywhere(&result.lower)
    } else {
        matches!(&result.upper, NullableRowAddrMask::AllowList(set) if has_no_true_rows(set))
    }
}

/// Drive `fut` while continuing to poll `in_flight` whenever `fut` yields
/// `Poll::Pending`, buffering any sibling futures that finish in the meantime.
///
/// Prevents cooperative-polling deadlocks when `reconcile_eval_domains` awaits a
/// `slot.row_addr` `OnceCell` whose leader future resides inside `in_flight`.
async fn drive_with_in_flight<T, Item, F, St>(
    fut: F,
    in_flight: &mut FuturesUnordered<St>,
    ready_buf: &mut VecDeque<Item>,
) -> T
where
    F: std::future::Future<Output = T>,
    St: std::future::Future<Output = Item>,
{
    tokio::pin!(fut);
    std::future::poll_fn(|cx| {
        if let std::task::Poll::Ready(out) = fut.as_mut().poll(cx) {
            return std::task::Poll::Ready(out);
        }
        if !in_flight.is_empty() {
            while let std::task::Poll::Ready(Some(item)) = in_flight.poll_next_unpin(cx) {
                ready_buf.push_back(item);
            }
            return fut.as_mut().poll(cx);
        }
        std::task::Poll::Pending
    })
    .await
}

async fn eval_or(
    expr: &ScalarIndexExpr,
    ctx: &EvalContext,
    index_loader: &dyn ScalarIndexLoader,
    metrics: &dyn MetricsCollector,
    track_nulls: bool,
) -> Result<EvalValue> {
    let mut branches = Vec::new();
    collect_or_branches(expr, &mut branches);

    let mut pending_recheck_count = branches.iter().filter(|b| b.needs_recheck()).count();
    let total = branches.len();
    let mut next_idx = 0usize;
    let mut in_flight = FuturesUnordered::new();

    let make_branch_fut = |idx: usize| {
        let branch = branches.get(idx).copied().ok_or_else(|| {
            Error::internal(format!(
                "OR branch index {idx} out of bounds for {total} branches"
            ))
        })?;
        Ok::<_, Error>(async move {
            let res = eval_node(branch, ctx, index_loader, metrics, track_nulls, true).await;
            (idx, res)
        })
    };

    let spawn_count = total.min(MAX_OR_CONCURRENCY);
    while next_idx < spawn_count {
        in_flight.push(make_branch_fut(next_idx)?);
        next_idx += 1;
    }

    let mut acc: Option<EvalValue> = None;
    let mut ready_buf = VecDeque::new();
    loop {
        let next_item = match ready_buf.pop_front() {
            Some(item) => Some(item),
            None => in_flight.next().await,
        };
        let Some((idx, res)) = next_item else {
            break;
        };
        let val = res?;
        if branches.get(idx).is_some_and(|b| b.needs_recheck()) {
            pending_recheck_count = pending_recheck_count.saturating_sub(1);
        }
        acc = Some(match acc {
            None => val,
            Some(prev) => {
                let (prev, val) = drive_with_in_flight(
                    reconcile_eval_domains(prev, val, ctx, index_loader),
                    &mut in_flight,
                    &mut ready_buf,
                )
                .await?;
                prev.bit_or(val)
            }
        });

        if let Some(current) = acc.as_ref()
            && is_all_true(current.as_result())
        {
            let is_exact = current.as_result().is_exact() && pending_recheck_count == 0;
            let domain = current.as_result().domain;
            return Ok(EvalValue::Owned(if is_exact {
                NullableIndexExprResult::exact(NullableRowAddrMask::all_rows()).with_domain(domain)
            } else {
                NullableIndexExprResult::at_least(NullableRowAddrMask::all_rows())
                    .with_domain(domain)
            }));
        }

        while next_idx < total && in_flight.len() + ready_buf.len() < MAX_OR_CONCURRENCY {
            in_flight.push(make_branch_fut(next_idx)?);
            next_idx += 1;
        }
    }

    acc.ok_or_else(|| Error::internal("OR expression produced no branches"))
}

/// Fold one completed conjunct into `acc`, returning `Some(short_circuit)` when
/// the conjunction can no longer match any row (preserving NULLs if `track_nulls`).
async fn fold_and_conjunct(
    acc: &mut Option<EvalValue>,
    pending_recheck_count: &mut usize,
    needs_recheck: bool,
    val: EvalValue,
    ctx: &EvalContext,
    index_loader: &dyn ScalarIndexLoader,
    track_nulls: bool,
) -> Result<Option<EvalValue>> {
    if needs_recheck {
        *pending_recheck_count = pending_recheck_count.saturating_sub(1);
    }
    let current = match acc.take() {
        None => val,
        Some(prev) => {
            let (prev, val) = reconcile_eval_domains(prev, val, ctx, index_loader).await?;
            prev.bit_and(val)
        }
    };
    if can_short_circuit_and(current.as_result(), track_nulls) {
        let is_exact = current.as_result().is_exact() && *pending_recheck_count == 0;
        let domain = current.as_result().domain;
        return Ok(Some(EvalValue::Owned(if is_exact {
            NullableIndexExprResult::exact(NullableRowAddrMask::allow_nothing()).with_domain(domain)
        } else {
            NullableIndexExprResult::at_most(NullableRowAddrMask::allow_nothing())
                .with_domain(domain)
        })));
    }
    *acc = Some(current);
    Ok(None)
}

async fn eval_and(
    expr: &ScalarIndexExpr,
    ctx: &EvalContext,
    index_loader: &dyn ScalarIndexLoader,
    metrics: &dyn MetricsCollector,
    track_nulls: bool,
    in_disjunction: bool,
) -> Result<EvalValue> {
    let mut conjuncts = Vec::new();
    collect_and_conjuncts(expr, &mut conjuncts);
    conjuncts.sort_by_key(|c| conjunct_priority(c, ctx, track_nulls));

    let mut pending_recheck_count = conjuncts.iter().filter(|c| c.needs_recheck()).count();
    let total = conjuncts.len();
    let mut next_idx = 0usize;
    let mut acc: Option<EvalValue> = None;

    // Phase 1 (Zero-I/O Ready CSE Drain): fold any already-populated CSE leaves
    // synchronously before issuing uncached I/O.
    while let Some(conjunct) = conjuncts.get(next_idx) {
        let ScalarIndexExpr::Query(search) = conjunct else {
            break;
        };
        if !ctx.is_ready(search, track_nulls) {
            break;
        };
        let val = eval_query(search, ctx, index_loader, metrics, track_nulls).await?;
        next_idx += 1;
        if let Some(short_circuit) = fold_and_conjunct(
            &mut acc,
            &mut pending_recheck_count,
            conjunct.needs_recheck(),
            val,
            ctx,
            index_loader,
            track_nulls,
        )
        .await?
        {
            return Ok(short_circuit);
        }
    }

    // Phase 2 (Initial Selective Probe): evaluate the highest-priority remaining
    // conjunct alone so selective or empty predicates short-circuit before tail I/O.
    if let Some(probe) = conjuncts.get(next_idx) {
        let val = eval_node(
            probe,
            ctx,
            index_loader,
            metrics,
            track_nulls,
            in_disjunction,
        )
        .await?;
        next_idx += 1;
        if let Some(short_circuit) = fold_and_conjunct(
            &mut acc,
            &mut pending_recheck_count,
            probe.needs_recheck(),
            val,
            ctx,
            index_loader,
            track_nulls,
        )
        .await?
        {
            return Ok(short_circuit);
        }
    }

    // Phase 3 (Bounded-Concurrency Tail Overlap): overlap remaining tail conjuncts
    // with bounded concurrency, dropping in-flight futures on error or short-circuit.
    let remaining = total.saturating_sub(next_idx);
    if remaining == 0 {
        return acc.ok_or_else(|| Error::internal("AND expression produced no conjuncts"));
    }
    if remaining == 1 {
        let tail = conjuncts.get(next_idx).ok_or_else(|| {
            Error::internal(format!(
                "AND conjunct index {next_idx} out of bounds for {total} conjuncts"
            ))
        })?;
        let val = eval_node(
            tail,
            ctx,
            index_loader,
            metrics,
            track_nulls,
            in_disjunction,
        )
        .await?;
        if let Some(short_circuit) = fold_and_conjunct(
            &mut acc,
            &mut pending_recheck_count,
            tail.needs_recheck(),
            val,
            ctx,
            index_loader,
            track_nulls,
        )
        .await?
        {
            return Ok(short_circuit);
        }
        return acc.ok_or_else(|| Error::internal("AND expression produced no conjuncts"));
    }

    let max_concurrency = if in_disjunction {
        MAX_NESTED_AND_CONCURRENCY
    } else {
        MAX_AND_CONCURRENCY
    };
    let spawn_count = remaining.min(max_concurrency);
    let mut in_flight = FuturesUnordered::new();
    let make_conjunct_fut = |idx: usize| {
        let conjunct = conjuncts.get(idx).copied().ok_or_else(|| {
            Error::internal(format!(
                "AND conjunct index {idx} out of bounds for {total} conjuncts"
            ))
        })?;
        Ok::<_, Error>(async move {
            let res = eval_node(
                conjunct,
                ctx,
                index_loader,
                metrics,
                track_nulls,
                in_disjunction,
            )
            .await;
            (idx, res)
        })
    };

    // Push in reverse order so FuturesUnordered's LIFO ready list polls
    // higher-priority conjuncts first.
    for idx in (next_idx..next_idx + spawn_count).rev() {
        in_flight.push(make_conjunct_fut(idx)?);
    }
    next_idx += spawn_count;

    let mut ready_buf = VecDeque::new();
    loop {
        let next_item = match ready_buf.pop_front() {
            Some(item) => Some(item),
            None => in_flight.next().await,
        };
        let Some((idx, res)) = next_item else {
            break;
        };
        let val = res?;
        let needs_recheck = conjuncts.get(idx).is_some_and(|c| c.needs_recheck());
        if let Some(short_circuit) = drive_with_in_flight(
            fold_and_conjunct(
                &mut acc,
                &mut pending_recheck_count,
                needs_recheck,
                val,
                ctx,
                index_loader,
                track_nulls,
            ),
            &mut in_flight,
            &mut ready_buf,
        )
        .await?
        {
            return Ok(short_circuit);
        }
        while next_idx < total && in_flight.len() + ready_buf.len() < max_concurrency {
            in_flight.push(make_conjunct_fut(next_idx)?);
            next_idx += 1;
        }
    }

    acc.ok_or_else(|| Error::internal("AND expression produced no conjuncts"))
}

#[cfg(test)]
mod tests;
