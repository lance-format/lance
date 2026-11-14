// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Per-segment coverage used to remove stale index hits at query time.

use std::sync::Arc;

use arrow::compute::filter_record_batch;
use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{BooleanArray, RecordBatch};
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use lance_core::deepsize::Context;
use lance_core::utils::address::RowAddress;
use lance_core::{Error, Result};
use lance_index::scalar::SearchResult;
use lance_select::{NullableRowAddrSet, RowAddrMask, RowAddrTreeMap, RowSetOps};
use lance_table::format::IndexMetadata;
use roaring::RoaringBitmap;
use tokio::sync::OnceCell;

use crate::Dataset;
use crate::index::frag_reuse::resolve_query_segment_coverages;
use crate::index::prefilter::DatasetPreFilter;

/// The row domain produced by an index segment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SegmentRowDomain {
    /// Row ids. Under stable row ids an update can move a row to a fragment another segment
    /// covers while it keeps its id, so a hit must name a live row stored in the segment's
    /// fragments.
    RowId,
    /// Physical row addresses. An in-place column rewrite keeps a row's address but takes its
    /// fragment out of the segment's coverage, so a hit must lie in a fragment the segment
    /// still covers.
    RowAddress,
}

/// The current fragment coverage of each segment in a logical index.
///
/// A segment's files can hold entries for rows it no longer answers for: an in-place column
/// rewrite takes a fragment out of the segment's coverage and leaves its old entries behind,
/// and under stable row ids an update can move a row into a fragment another segment covers.
/// The scan bounds index answers only by the union of every segment's coverage, so once
/// another segment covers such a fragment, each segment's hits must be limited to its own.
#[derive(Debug)]
pub struct SegmentCoverage {
    /// Current fragment coverage in segment order.
    fragments: Vec<RoaringBitmap>,
    restriction: Restriction,
}

/// How [`SegmentCoverage`] limits a segment's hits.
#[derive(Debug)]
enum Restriction {
    /// Stable row ids: the live rows currently stored in the segment's fragments.
    LiveRowIds(Arc<Dataset>),
    /// Row addresses: the segment's fragments.
    Fragments,
}

/// The rows one segment may report.
enum AllowedRows<'a> {
    LiveRowIds(Arc<RowAddrMask>),
    Fragments(&'a RoaringBitmap),
}

impl AllowedRows<'_> {
    fn contains(&self, row: u64) -> bool {
        match self {
            Self::LiveRowIds(allowed) => allowed.selected(row),
            Self::Fragments(fragments) => {
                fragments.contains(RowAddress::new_from_u64(row).fragment_id())
            }
        }
    }

    fn restrict(&self, rows: &mut RowAddrTreeMap) {
        match self {
            Self::LiveRowIds(allowed) => rows.mask(allowed),
            // Walk the hits, not the coverage: the coverage can span the whole table, and
            // `MapIndexExec` searches for every input batch.
            Self::Fragments(fragments) => {
                let covered = rows
                    .iter()
                    .map(|(fragment, _)| *fragment)
                    .filter(|fragment| fragments.contains(*fragment))
                    .collect::<Vec<_>>();
                rows.retain_fragments(covered);
            }
        }
    }
}

impl SegmentCoverage {
    /// Resolve each segment's current coverage when its hits need limiting.
    ///
    /// `None` for a single segment, whose coverage is the union the scan already applies, and
    /// for row-id hits on a table without stable row ids. Those ids are addresses:
    /// `LogicalScalarIndex` resolves them as [`SegmentRowDomain::RowAddress`], while FTS
    /// still leaves them unrestricted (#6283).
    pub(crate) async fn resolve(
        dataset: &Dataset,
        segments: &[&IndexMetadata],
        row_domain: SegmentRowDomain,
    ) -> Result<Option<Self>> {
        if segments.len() <= 1 {
            return Ok(None);
        }
        let restriction = match row_domain {
            SegmentRowDomain::RowAddress => Restriction::Fragments,
            SegmentRowDomain::RowId if dataset.manifest.uses_stable_row_ids() => {
                Restriction::LiveRowIds(Arc::new(dataset.clone()))
            }
            SegmentRowDomain::RowId => return Ok(None),
        };

        let fragments = resolve_query_segment_coverages(dataset, segments).await?;
        Ok(Some(Self {
            fragments,
            restriction,
        }))
    }

    fn coverage(&self, index: usize) -> Result<&RoaringBitmap> {
        self.fragments.get(index).ok_or_else(|| {
            Error::internal(format!(
                "segment coverage index {index} is out of bounds for {} segments",
                self.fragments.len()
            ))
        })
    }

    /// The live stable row ids currently stored in segment `index`'s coverage.
    ///
    /// `DatasetPreFilter` caches the allow-list by dataset version and coverage. Callers keep
    /// loading it lazily so a segment with no hits does not read row-id sequences or deletion
    /// vectors.
    pub(crate) async fn live_rows(&self, index: usize) -> Result<Arc<RowAddrMask>> {
        let Restriction::LiveRowIds(dataset) = &self.restriction else {
            return Err(Error::internal(
                "segment coverage of row-address hits has no live row ids",
            ));
        };
        DatasetPreFilter::do_create_deletion_mask_row_id(
            dataset.clone(),
            Some(self.coverage(index)?.clone()),
        )
        .await
    }

    async fn allowed_rows(&self, index: usize) -> Result<AllowedRows<'_>> {
        Ok(match &self.restriction {
            Restriction::LiveRowIds(_) => AllowedRows::LiveRowIds(self.live_rows(index).await?),
            Restriction::Fragments => AllowedRows::Fragments(self.coverage(index)?),
        })
    }

    /// Drop the hits of segment `index` that lie outside its coverage.
    pub(crate) async fn restrict(
        &self,
        index: usize,
        result: SearchResult,
    ) -> Result<SearchResult> {
        let set = result.row_addrs();
        if set.selected_rows().is_empty() && set.null_rows().is_empty() {
            return Ok(result);
        }
        let allowed = self.allowed_rows(index).await?;
        let restrict = |set: NullableRowAddrSet| {
            let (mut selected, mut nulls) = set.into_parts();
            allowed.restrict(&mut selected);
            allowed.restrict(&mut nulls);
            NullableRowAddrSet::new(selected, nulls)
        };
        Ok(match result {
            SearchResult::Exact(set) => SearchResult::Exact(restrict(set)),
            SearchResult::AtMost(set) => SearchResult::AtMost(restrict(set)),
            SearchResult::AtLeast(set) => SearchResult::AtLeast(restrict(set)),
        })
    }

    /// Drop lookup matches of segment `index` that lie outside its coverage. `row_column`
    /// names the matched-row column of the lookup batches.
    pub(crate) fn restrict_lookup<'a>(
        &'a self,
        index: usize,
        row_column: &'a str,
        matches: BoxStream<'a, Result<RecordBatch>>,
    ) -> BoxStream<'a, Result<RecordBatch>> {
        let allowed = Arc::new(OnceCell::new());
        matches
            .and_then(move |batch| {
                let allowed = allowed.clone();
                async move {
                    let allowed = allowed.get_or_try_init(|| self.allowed_rows(index)).await?;
                    let rows = batch.column_by_name(row_column).ok_or_else(|| {
                        Error::internal(format!(
                            "scalar index lookup batch is missing the {row_column} column"
                        ))
                    })?;
                    let keep = rows
                        .as_primitive::<UInt64Type>()
                        .values()
                        .iter()
                        .map(|&row| allowed.contains(row))
                        .collect::<BooleanArray>();
                    Ok(filter_record_batch(&batch, &keep)?)
                }
            })
            .try_filter(|batch| std::future::ready(batch.num_rows() > 0))
            .boxed()
    }

    pub(crate) fn deep_size_of_children(&self, _context: &mut Context) -> usize {
        self.fragments
            .iter()
            .map(|fragments| fragments.serialized_size())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{RecordBatchIterator, record_batch};
    use rstest::rstest;
    use uuid::Uuid;

    use super::*;
    use crate::dataset::WriteParams;

    fn segment(fragment_bitmap: Option<RoaringBitmap>) -> IndexMetadata {
        IndexMetadata {
            uuid: Uuid::new_v4(),
            name: "idx".to_string(),
            fields: vec![0],
            covering_fields: vec![],
            dataset_version: 1,
            fragment_bitmap,
            index_details: None,
            index_version: 0,
            created_at: None,
            base_id: None,
            files: None,
            clustering_id: None,
            metadata: Default::default(),
        }
    }

    async fn dataset(enable_stable_row_ids: bool) -> Dataset {
        let batch = record_batch!(("value", Int32, [1])).unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        Dataset::write(
            reader,
            "memory://",
            Some(WriteParams {
                enable_stable_row_ids,
                ..Default::default()
            }),
        )
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn skips_masking_for_one_segment_or_row_ids_without_stable_row_ids() {
        let stable = dataset(true).await;
        let single = segment(None);
        for row_domain in [SegmentRowDomain::RowId, SegmentRowDomain::RowAddress] {
            assert!(
                SegmentCoverage::resolve(&stable, &[&single], row_domain)
                    .await
                    .unwrap()
                    .is_none()
            );
        }

        let first = segment(None);
        let second = segment(None);
        let plain = dataset(false).await;
        assert!(
            SegmentCoverage::resolve(&plain, &[&first, &second], SegmentRowDomain::RowId)
                .await
                .unwrap()
                .is_none()
        );
    }

    #[rstest]
    #[case::row_ids(true, SegmentRowDomain::RowId)]
    #[case::row_addresses(true, SegmentRowDomain::RowAddress)]
    #[case::row_addresses_without_stable_row_ids(false, SegmentRowDomain::RowAddress)]
    #[tokio::test]
    async fn refuses_unknown_coverage_when_masking_is_required(
        #[case] stable_row_ids: bool,
        #[case] row_domain: SegmentRowDomain,
    ) {
        let dataset = dataset(stable_row_ids).await;
        let first = segment(None);
        let second = segment(Some(RoaringBitmap::from_iter([0])));
        let error = SegmentCoverage::resolve(&dataset, &[&first, &second], row_domain)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error.to_string().contains("missing fragment coverage"),
            "{error}"
        );

        let first = segment(Some(RoaringBitmap::from_iter([0])));
        let second = segment(Some(RoaringBitmap::from_iter([0])));
        let coverage = SegmentCoverage::resolve(&dataset, &[&first, &second], row_domain)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            coverage.fragments,
            vec![RoaringBitmap::from_iter([0]), RoaringBitmap::from_iter([0])]
        );
    }
}
