// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashSet;
use std::sync::Arc;

use arrow_schema::{DataType, Schema as ArrowSchema};
use async_recursion::async_recursion;
use datafusion::common::DFSchema;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion_expr::utils::{conjunction, split_conjunction_owned};
use datafusion_expr::{Expr, ExprSchemable, Volatility};
use lance_core::{Error, Result};
use lance_datafusion::planner::Planner;
use lance_file::version::ConcreteFileVersion;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::scalar::expression::{ScalarIndexExpr, ScalarIndexSearch, create_filter_plan};
use lance_table::format::IndexMetadata;
use lance_table::format::overlay::staleness::{overlaid_fragments, overlay_affects_index};
use roaring::RoaringBitmap;

use super::Dataset;
use super::scanner::ExprFilter;
use crate::index::DatasetIndexExt;
use crate::index::frag_reuse::resolve_query_segment_coverages;
use crate::index::loading::DatasetIndexInternalExt;
use crate::index::scalar::IndexDetails;

impl Dataset {
    /// Return the physical fragment IDs that may contain rows matching `filter`
    /// in this dataset snapshot.
    ///
    /// This reads index and dataset metadata without scanning data files. It
    /// retains fragments that lack usable index coverage or cannot be ruled out.
    /// Apply the original filter to the candidates using the same snapshot.
    /// Predicates with query-scoped or volatile functions and segments that
    /// require row-level address translation are handled conservatively.
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # use lance::dataset::scanner::ExprFilter;
    /// # async fn example(dataset: &Dataset) -> Result<()> {
    /// let filter = ExprFilter::Sql("price > 100".into());
    /// let candidates = dataset.candidate_fragment_ids(filter).await?;
    /// for id in candidates {
    ///     // Schedule a scan of this fragment in the current dataset version.
    ///     let _ = id;
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn candidate_fragment_ids(&self, filter: ExprFilter) -> Result<RoaringBitmap> {
        let filter_schema = self.scan().filterable_schema()?;
        let schema = Arc::new(ArrowSchema::from(filter_schema.as_ref()));
        let expr = filter.to_unoptimized_datafusion(self.schema(), &filter_schema)?;
        let df_schema = DFSchema::try_from(schema.clone())?;
        if expr.get_type(&df_schema)? != DataType::Boolean {
            return Err(Error::invalid_input(format!(
                "The filter {expr} does not return a boolean"
            )));
        }
        let all_fragments = self.fragment_bitmap.as_ref();
        if all_fragments.is_empty()
            || self.manifest.data_storage_format.lance_file_format() == ConcreteFileVersion::V1
        {
            return Ok(all_fragments.clone());
        }

        // Query-scoped functions such as now() may be evaluated again on workers.
        // Discard their conjuncts before the optimizer can turn them into literals.
        let mut immutable_conjuncts = Vec::new();
        for conjunct in split_conjunction_owned(expr) {
            if is_immutable(&conjunct)? {
                immutable_conjuncts.push(conjunct);
            }
        }
        let Some(expr) = conjunction(immutable_conjuncts) else {
            return Ok(all_fragments.clone());
        };
        let expr = Planner::new(schema).optimize_expr(expr)?;

        let indices = self.load_indices().await?;
        let pruning_indices: Vec<_> = indices
            .iter()
            .filter(|index| {
                index.fragment_bitmap.is_some()
                    && index.index_details.as_ref().is_some_and(|details| {
                        IndexDetails(details.clone())
                            .get_plugin()
                            .is_ok_and(|plugin| plugin.supports_fragment_pruning())
                    })
            })
            .cloned()
            .collect();
        let index_info = self
            .scalar_index_info_for_segments_ignoring(&pruning_indices, &HashSet::new())
            .await?;
        let Some(query) = create_filter_plan(expr, &index_info, true)?.index_query else {
            return Ok(all_fragments.clone());
        };
        self.evaluate_fragment_query(&query, &pruning_indices).await
    }

    #[async_recursion]
    async fn evaluate_fragment_query(
        &self,
        query: &ScalarIndexExpr,
        indices: &[IndexMetadata],
    ) -> Result<RoaringBitmap> {
        Ok(match query {
            ScalarIndexExpr::Query(search) => {
                self.fragment_candidates_for_search(search, indices).await?
            }
            ScalarIndexExpr::And(left, right) => {
                let left = self.evaluate_fragment_query(left, indices).await?;
                if left.is_empty() {
                    return Ok(left);
                }
                left & self.evaluate_fragment_query(right, indices).await?
            }
            ScalarIndexExpr::Or(left, right) => {
                let left = self.evaluate_fragment_query(left, indices).await?;
                if &left == self.fragment_bitmap.as_ref() {
                    return Ok(left);
                }
                left | self.evaluate_fragment_query(right, indices).await?
            }
            // A fragment may contain both matching and nonmatching rows.
            ScalarIndexExpr::Not(_) => self.fragment_bitmap.as_ref().clone(),
        })
    }

    async fn fragment_candidates_for_search(
        &self,
        search: &ScalarIndexSearch,
        indices: &[IndexMetadata],
    ) -> Result<RoaringBitmap> {
        let segments: Vec<_> = indices
            .iter()
            .filter(|index| index.name == search.index_name)
            .collect();
        let all_fragments = self.fragment_bitmap.as_ref();
        let overlaid = overlaid_fragments(self.fragments());
        let mut covered = RoaringBitmap::new();
        let mut hits = RoaringBitmap::new();
        let mut unknown = RoaringBitmap::new();
        for (segment, coverage) in segments
            .iter()
            .zip(resolve_query_segment_coverages(self, &segments).await?)
        {
            let coverage = coverage & all_fragments;
            let mut fresh_coverage = coverage;
            let fields = segment.dependency_fields();
            for (&id, fragment) in &overlaid {
                if fresh_coverage.contains(id)
                    && fragment.overlays.iter().any(|overlay| {
                        overlay_affects_index(
                            overlay,
                            &fields,
                            segment.dataset_version,
                            self.schema(),
                        )
                    })
                {
                    fresh_coverage.remove(id);
                    unknown.insert(id);
                }
            }
            if fresh_coverage.is_empty() {
                continue;
            }
            let index = self
                .open_scalar_index(&search.column, &segment.uuid, &NoOpMetricsCollector)
                .await?;
            if let Some(matches) = index
                .candidate_fragment_ids(search.query.as_ref(), &NoOpMetricsCollector)
                .await?
            {
                hits |= matches & &fresh_coverage;
                covered |= fresh_coverage;
            } else {
                unknown |= fresh_coverage;
            }
        }
        Ok((all_fragments - &covered) | hits | unknown)
    }
}

fn is_immutable(expr: &Expr) -> Result<bool> {
    let mut immutable = true;
    expr.apply(|node| {
        if let Expr::ScalarFunction(call) = node
            && call.func.signature().volatility != Volatility::Immutable
        {
            immutable = false;
            return Ok(TreeNodeRecursion::Stop);
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(immutable)
}

#[cfg(test)]
mod tests {
    use arrow_array::{
        ArrayRef, Int32Array, RecordBatch, RecordBatchIterator, TimestampMillisecondArray,
        UInt64Array, record_batch,
    };
    use lance_core::ROW_ADDR;
    use lance_core::utils::address::RowAddress;
    use lance_index::IndexType;
    use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams};
    use rstest::rstest;

    use super::*;
    use crate::dataset::WriteParams;
    use crate::dataset::optimize::{CompactionOptions, compact_files};

    async fn write_dataset(stable_row_ids: bool) -> Dataset {
        let batch = record_batch!(
            (
                "price",
                Int32,
                [Some(0), None, Some(10), Some(11), Some(20), Some(21)]
            ),
            ("user_id", Int32, [1, 1, 1, 1, 2, 2])
        )
        .unwrap();
        let reader = RecordBatchIterator::new([Ok(batch.clone())], batch.schema());
        Dataset::write(
            reader,
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 2,
                max_rows_per_group: 2,
                enable_stable_row_ids: stable_row_ids,
                ..Default::default()
            }),
        )
        .await
        .unwrap()
    }

    async fn build_segments(
        dataset: &mut Dataset,
        column: &str,
        kind: BuiltinIndexType,
        fragment_groups: &[&[u32]],
    ) {
        let name = format!("{column}_index");
        let force_multiple_zones = kind == BuiltinIndexType::ZoneMap;
        let mut params = ScalarIndexParams::for_builtin(kind);
        if force_multiple_zones {
            params = params.with_params(&serde_json::json!({"rows_per_zone": 1}));
        }
        let mut segments = Vec::new();
        for group in fragment_groups {
            segments.push(
                dataset
                    .create_index_builder(&[column], IndexType::Scalar, &params)
                    .name(name.clone())
                    .fragments(group.to_vec())
                    .execute_uncommitted()
                    .await
                    .unwrap(),
            );
        }
        dataset
            .commit_existing_index_segments(&name, column, segments)
            .await
            .unwrap();
    }

    async fn assert_candidates(dataset: &Dataset, filter: &str, expected: &[u32]) {
        let candidates = dataset
            .candidate_fragment_ids(ExprFilter::Sql(filter.into()))
            .await
            .unwrap();
        assert_eq!(candidates, expected.iter().copied().collect(), "{filter}");

        let matching_addresses = |scanner: &mut super::super::scanner::Scanner| {
            scanner
                .use_scalar_index(false)
                .with_row_address()
                .filter(filter)
                .unwrap();
        };
        let mut full_scan = dataset.scan();
        matching_addresses(&mut full_scan);
        let full = full_scan.try_into_batch().await.unwrap();
        let full_addrs = full[ROW_ADDR]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert!(
            full_addrs
                .values()
                .iter()
                .all(|&addr| { candidates.contains(RowAddress::from(addr).fragment_id()) })
        );
        let mut candidate_scan = dataset.scan();
        matching_addresses(&mut candidate_scan);
        candidate_scan.with_fragments(
            dataset
                .fragments()
                .iter()
                .filter(|fragment| candidates.contains(fragment.id as u32))
                .cloned()
                .collect(),
        );
        let candidate_count = candidate_scan.count_rows().await.unwrap();
        assert_eq!(candidate_count as usize, full.num_rows());
    }

    #[rstest]
    #[tokio::test]
    async fn test_fragment_pruning_partial_coverage(#[values(false, true)] stable_row_ids: bool) {
        let mut dataset = write_dataset(stable_row_ids).await;
        assert_candidates(&dataset, "price >= 10", &[0, 1, 2]).await;
        build_segments(
            &mut dataset,
            "price",
            BuiltinIndexType::ZoneMap,
            &[&[0], &[1]],
        )
        .await;
        build_segments(
            &mut dataset,
            "user_id",
            BuiltinIndexType::BloomFilter,
            &[&[1, 2]],
        )
        .await;

        assert_candidates(&dataset, "price >= 10 AND user_id = 2", &[2]).await;
        assert_candidates(&dataset, "price >= 10 OR user_id = 2", &[0, 1, 2]).await;
        assert_candidates(&dataset, "price > 100 OR user_id = 2", &[0, 2]).await;
        assert_candidates(&dataset, "price >= 10 AND price + user_id > 10", &[1, 2]).await;
        assert_candidates(&dataset, "price >= 10 OR price + user_id > 10", &[0, 1, 2]).await;
        assert_candidates(&dataset, "price IS NULL", &[0, 2]).await;
        assert_candidates(&dataset, "NOT (price IS NULL)", &[0, 1, 2]).await;
    }

    #[tokio::test]
    async fn test_fragment_pruning_after_append_only_reads_index_files() {
        let mut dataset = write_dataset(false).await;
        build_segments(
            &mut dataset,
            "price",
            BuiltinIndexType::ZoneMap,
            &[&[0, 1, 2]],
        )
        .await;
        let batch = record_batch!(("price", Int32, [Some(100)]), ("user_id", Int32, [3])).unwrap();
        dataset
            .append(
                RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
                None,
            )
            .await
            .unwrap();

        dataset.object_store.io_stats_incremental();
        let candidates = dataset
            .candidate_fragment_ids(ExprFilter::Sql("price > 90".into()))
            .await
            .unwrap();
        assert_eq!(candidates, RoaringBitmap::from_iter([3]));
        let stats = dataset.object_store.io_stats_incremental();
        for request in &stats.requests {
            assert!(
                !request.path.parts().any(|part| part.as_ref() == "data"),
                "fragment pruning opened data: {request:?}"
            );
        }
        assert_candidates(&dataset, "price > 90", &[3]).await;

        let error = dataset
            .candidate_fragment_ids(ExprFilter::Sql("price".into()))
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("does not return a boolean"));
    }

    #[tokio::test]
    async fn test_fragment_pruning_accepts_scanner_system_columns() {
        let mut dataset = write_dataset(false).await;
        build_segments(
            &mut dataset,
            "price",
            BuiltinIndexType::ZoneMap,
            &[&[0, 1, 2]],
        )
        .await;
        let address = u64::from(RowAddress::new_from_parts(1, 0));
        assert_candidates(
            &dataset,
            &format!("price >= 10 AND _rowaddr = {address}"),
            &[1, 2],
        )
        .await;
    }

    #[tokio::test]
    async fn test_fragment_pruning_skips_query_scoped_and_volatile_predicates() {
        let batch = RecordBatch::try_from_iter([
            ("price", Arc::new(Int32Array::from(vec![0, 10])) as ArrayRef),
            (
                "event_ts",
                Arc::new(TimestampMillisecondArray::from(vec![
                    4_102_444_800_000,
                    4_102_531_200_000,
                ])) as ArrayRef,
            ),
        ])
        .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 1,
                max_rows_per_group: 1,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        build_segments(&mut dataset, "price", BuiltinIndexType::ZoneMap, &[&[0, 1]]).await;
        build_segments(
            &mut dataset,
            "event_ts",
            BuiltinIndexType::ZoneMap,
            &[&[0, 1]],
        )
        .await;

        // A literal time permits pruning. A query-scoped time can differ across
        // the driver and workers, so only the independent price conjunct can prune.
        assert_candidates(&dataset, "event_ts <= TIMESTAMP '2099-12-31 00:00:00'", &[]).await;
        assert_candidates(
            &dataset,
            "event_ts <= TIMESTAMP '2100-01-01 00:00:00'",
            &[0],
        )
        .await;
        for (filter, expected) in [
            ("event_ts <= now()", &[0, 1][..]),
            ("price >= 10 AND event_ts <= now()", &[1]),
            ("price >= 10 OR event_ts <= now()", &[0, 1]),
            ("price >= 10 AND random() < 0.5", &[1]),
        ] {
            let candidates = dataset
                .candidate_fragment_ids(ExprFilter::Sql(filter.into()))
                .await
                .unwrap();
            assert_eq!(
                candidates,
                RoaringBitmap::from_iter(expected.iter().copied()),
                "{filter}"
            );
        }
    }

    #[tokio::test]
    async fn test_fragment_pruning_after_compaction() {
        let mut dataset = write_dataset(false).await;
        build_segments(
            &mut dataset,
            "price",
            BuiltinIndexType::ZoneMap,
            &[&[0, 1, 2]],
        )
        .await;
        let snapshot = dataset.clone();

        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 6,
                defer_index_remap: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        assert_eq!(dataset.fragments().len(), 1);
        let current_id = dataset.fragments()[0].id as u32;
        assert_ne!(current_id, 0);
        assert_candidates(&dataset, "price = 0", &[current_id]).await;
        assert_candidates(&dataset, "price > 100", &[current_id]).await;
        assert_candidates(&snapshot, "price = 0", &[0]).await;
    }
}
