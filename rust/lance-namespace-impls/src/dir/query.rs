// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow::array::Float32Array;
use lance::dataset::scanner::Scanner;
use lance_core::{Error, Result};
use lance_index::scalar::inverted::query::{
    BooleanQuery, BoostQuery, FtsQuery, MatchQuery, MultiMatchQuery, Occur, Operator, PhraseQuery,
};
use lance_namespace::error::NamespaceError;
use lance_namespace::models::{QueryTableRequestColumns, QueryTableRequestVector};

use super::DirectoryNamespace;

pub(super) fn apply_probe_bounds(
    scanner: &mut Scanner,
    nprobes: Option<i32>,
    minimum_nprobes: Option<i32>,
    maximum_nprobes: Option<i32>,
) -> Result<()> {
    let parse_probe_count = |name: &str, value: i32| {
        usize::try_from(value)
            .map_err(|_| Error::invalid_input(format!("{name} must be non-negative")))
    };

    let maximum_is_unbounded = maximum_nprobes == Some(0);
    if let Some(nprobes) = nprobes {
        let nprobes = parse_probe_count("nprobes", nprobes)?;
        if maximum_is_unbounded {
            scanner.minimum_nprobes(nprobes);
        } else {
            scanner.nprobes(nprobes);
        }
    }
    if let Some(minimum_nprobes) = minimum_nprobes {
        scanner.minimum_nprobes(parse_probe_count("minimum_nprobes", minimum_nprobes)?);
    }
    if let Some(maximum_nprobes) = maximum_nprobes {
        let maximum_nprobes = parse_probe_count("maximum_nprobes", maximum_nprobes)?;
        if maximum_nprobes != 0 {
            scanner.maximum_nprobes(maximum_nprobes);
        }
    }
    Ok(())
}

impl DirectoryNamespace {
    /// Apply all query parameters from a `QueryTableRequest`-like source onto a `Scanner`.
    ///
    /// This covers vector search, filters, column projection, limits, and ANN tuning knobs so
    /// that `explain_table_query_plan` / `analyze_table_query_plan` produce an accurate plan.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn apply_query_params_to_scanner(
        scanner: &mut Scanner,
        filter: Option<&str>,
        columns: Option<&QueryTableRequestColumns>,
        vector_column: Option<&str>,
        vector: &QueryTableRequestVector,
        k: i32,
        offset: Option<i32>,
        prefilter: Option<bool>,
        bypass_vector_index: Option<bool>,
        nprobes: Option<i32>,
        minimum_nprobes: Option<i32>,
        maximum_nprobes: Option<i32>,
        ef: Option<i32>,
        refine_factor: Option<i32>,
        distance_type: Option<&str>,
        fast_search_flag: Option<bool>,
        with_row_id: Option<bool>,
        lower_bound: Option<f32>,
        upper_bound: Option<f32>,
        operation: &str,
    ) -> Result<()> {
        if let Some(pf) = prefilter {
            scanner.prefilter(pf);
        }

        if let Some(filter) = filter {
            scanner.filter(filter).map_err(|e| {
                Error::invalid_input_source(
                    format!("Invalid filter expression for {}: {}", operation, e).into(),
                )
            })?;
        }

        if let Some(cols) = columns {
            if let Some(ref names) = cols.column_names {
                scanner.project(names.as_slice()).map_err(|e| {
                    Error::invalid_input_source(
                        format!("Invalid column projection for {}: {}", operation, e).into(),
                    )
                })?;
            } else if let Some(ref aliases) = cols.column_aliases {
                // aliases maps output_alias -> source_column
                let pairs: Vec<(&str, &str)> = aliases
                    .iter()
                    .map(|(alias, src)| (alias.as_str(), src.as_str()))
                    .collect();
                scanner.project_with_transform(&pairs).map_err(|e| {
                    Error::invalid_input_source(
                        format!("Invalid column aliases for {}: {}", operation, e).into(),
                    )
                })?;
            }
        }

        // Resolve query vector: prefer single_vector, fall back to first row of multi_vector.
        let query_vec: Option<Vec<f32>> = vector
            .single_vector
            .as_ref()
            .filter(|v| !v.is_empty())
            .cloned()
            .or_else(|| {
                vector
                    .multi_vector
                    .as_ref()
                    .and_then(|mv| mv.first())
                    .filter(|v| !v.is_empty())
                    .cloned()
            });

        if let Some(q_vec) = query_vec {
            let col = vector_column.unwrap_or("vector");
            let q = Arc::new(Float32Array::from(q_vec));
            scanner
                .nearest(col, q.as_ref(), k.max(1) as usize)
                .map_err(|e| {
                    Error::invalid_input_source(
                        format!("Invalid vector query for {}: {}", operation, e).into(),
                    )
                })?;

            // ANN parameters — must be applied after nearest().
            apply_probe_bounds(scanner, nprobes, minimum_nprobes, maximum_nprobes)?;
            if let Some(e) = ef {
                scanner.ef(e.max(1) as usize);
            }
            if let Some(rf) = refine_factor {
                scanner.refine(rf.max(0) as u32);
            }
            // bypass_vector_index and fast_search are mutually exclusive; apply in order.
            if let Some(true) = bypass_vector_index {
                scanner.use_index(false);
            }
            if let Some(true) = fast_search_flag {
                scanner.fast_search();
            }
            if lower_bound.is_some() || upper_bound.is_some() {
                scanner.distance_range(lower_bound, upper_bound);
            }
            if let Some(dt) = distance_type {
                let metric = Self::parse_metric_type(Some(dt))?;
                scanner.distance_metric(metric);
            }
            // Apply offset on top of the k nearest results.
            if let Some(off) = offset.filter(|&o| o > 0) {
                scanner.limit(None, Some(off as i64)).map_err(|e| {
                    Error::invalid_input_source(
                        format!("Invalid offset for {}: {}", operation, e).into(),
                    )
                })?;
            }
        } else {
            // Scalar (non-vector) query: treat k as a row LIMIT.
            let limit = if k > 0 { Some(k as i64) } else { None };
            scanner
                .limit(limit, offset.map(|o| o as i64))
                .map_err(|e| {
                    Error::invalid_input_source(
                        format!("Invalid limit/offset for {}: {}", operation, e).into(),
                    )
                })?;
        }

        if let Some(true) = with_row_id {
            scanner.with_row_id();
        }

        Ok(())
    }
}

/// Maps a namespace structured `FtsQuery` model into the engine `FtsQuery`. Mirrors the mapping the
/// JNI scanner performs, so the local `queryTable` path honors `structured_query` the same way a
/// `fragment.newScan(fullTextQuery)` does.
pub(super) fn build_engine_fts_query(
    query: &lance_namespace::models::FtsQuery,
) -> std::result::Result<FtsQuery, NamespaceError> {
    if let Some(ref m) = query.r#match {
        Ok(FtsQuery::Match(build_engine_match_query(m)?))
    } else if let Some(ref p) = query.phrase {
        let mut phrase = PhraseQuery::new(p.terms.clone());
        if let Some(ref column) = p.column {
            phrase = phrase.with_column(Some(column.clone()));
        }
        if let Some(slop) = p.slop {
            phrase = phrase.with_slop(slop as u32);
        }
        Ok(FtsQuery::Phrase(phrase))
    } else if let Some(ref mm) = query.multi_match {
        let match_queries = mm
            .match_queries
            .iter()
            .map(build_engine_match_query)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(FtsQuery::MultiMatch(MultiMatchQuery { match_queries }))
    } else if let Some(ref b) = query.boolean {
        let mut clauses: Vec<(Occur, FtsQuery)> = Vec::new();
        for clause in &b.must {
            clauses.push((Occur::Must, build_engine_fts_query(clause)?));
        }
        for clause in &b.should {
            clauses.push((Occur::Should, build_engine_fts_query(clause)?));
        }
        for clause in &b.must_not {
            clauses.push((Occur::MustNot, build_engine_fts_query(clause)?));
        }
        Ok(FtsQuery::Boolean(BooleanQuery::new(clauses)))
    } else if let Some(ref boost) = query.boost {
        let positive = build_engine_fts_query(&boost.positive)?;
        let negative = build_engine_fts_query(&boost.negative)?;
        Ok(FtsQuery::Boost(BoostQuery::new(
            positive,
            negative,
            boost.negative_boost,
        )))
    } else {
        Err(NamespaceError::InvalidInput {
            message: "structured_query.query must set exactly one of match, phrase, multi_match, \
                      boolean, or boost"
                .to_string(),
        })
    }
}

fn build_engine_match_query(
    m: &lance_namespace::models::MatchQuery,
) -> std::result::Result<MatchQuery, NamespaceError> {
    let mut match_query = MatchQuery::new(m.terms.clone());
    if let Some(ref column) = m.column {
        match_query = match_query.with_column(Some(column.clone()));
    }
    if let Some(boost) = m.boost {
        match_query = match_query.with_boost(boost);
    }
    if let Some(fuzziness) = m.fuzziness {
        match_query = match_query.with_fuzziness(Some(fuzziness as u32));
    }
    if let Some(max_expansions) = m.max_expansions {
        match_query = match_query.with_max_expansions(max_expansions as usize);
    }
    if let Some(ref operator) = m.operator {
        let op =
            Operator::try_from(operator.as_str()).map_err(|e| NamespaceError::InvalidInput {
                message: format!("Invalid FTS operator: {:?}", e),
            })?;
        match_query = match_query.with_operator(op);
    }
    if let Some(prefix_length) = m.prefix_length {
        match_query = match_query.with_prefix_length(prefix_length as u32);
    }
    Ok(match_query)
}
