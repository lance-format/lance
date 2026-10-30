// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Per-segment coverage used to remove stale row-ID hits at query time.

use std::sync::Arc;

use arrow::compute::filter_record_batch;
use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{BooleanArray, RecordBatch};
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use lance_core::deepsize::Context;
use lance_core::{Error, ROW_ID, Result};
use lance_index::scalar::SearchResult;
use lance_select::{NullableRowAddrSet, RowAddrMask, RowSetOps};
use lance_table::format::IndexMetadata;
use roaring::RoaringBitmap;
use tokio::sync::OnceCell;

use crate::Dataset;
use crate::index::frag_reuse::resolve_query_segment_coverages;
use crate::index::prefilter::DatasetPreFilter;

/// The row domain produced by an index segment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SegmentRowDomain {
    /// Stable row ids can remain live after moving out of a segment's coverage.
    RowId,
    /// Physical addresses stop being live when a row moves.
    RowAddress,
}

/// The current fragment coverage of each segment in a logical index.
///
/// Under stable row ids, a rewritten row keeps its id while moving to a fragment another
/// segment may cover. A stale posting in the old segment can therefore name a row id that is
/// still live elsewhere. Segment-local masking is only needed for row-ID segments on a
/// multi-segment stable-row-ID index: single-segment indexes have no competing segment, and an
/// address-domain hit for a moved row names the retired physical copy.
#[derive(Debug)]
pub struct SegmentCoverage {
    dataset: Arc<Dataset>,
    /// Current fragment coverage in segment order.
    fragments: Vec<RoaringBitmap>,
}

impl SegmentCoverage {
    /// Resolve query coverage when segment-local row-ID masking is required.
    ///
    /// `None` preserves the existing no-filter path for address-domain hits, tables without
    /// stable row ids, and single-segment indexes.
    pub(crate) async fn resolve(
        dataset: &Dataset,
        segments: &[&IndexMetadata],
        row_domain: SegmentRowDomain,
    ) -> Result<Option<Self>> {
        if row_domain == SegmentRowDomain::RowAddress
            || !dataset.manifest.uses_stable_row_ids()
            || segments.len() <= 1
        {
            return Ok(None);
        }

        let fragments = resolve_query_segment_coverages(dataset, segments).await?;
        Ok(Some(Self {
            dataset: Arc::new(dataset.clone()),
            fragments,
        }))
    }

    /// The live stable row ids currently stored in segment `index`'s coverage.
    ///
    /// `DatasetPreFilter` caches the allow-list by dataset version and coverage. Callers keep
    /// loading it lazily so a segment with no hits does not read row-id sequences or deletion
    /// vectors.
    pub(crate) async fn live_rows(&self, index: usize) -> Result<Arc<RowAddrMask>> {
        let fragments = self.fragments.get(index).ok_or_else(|| {
            Error::internal(format!(
                "segment coverage index {index} is out of bounds for {} segments",
                self.fragments.len()
            ))
        })?;
        DatasetPreFilter::do_create_deletion_mask_row_id(
            self.dataset.clone(),
            Some(fragments.clone()),
        )
        .await
    }

    /// Drop the hits of segment `index` that are not live rows of the fragments it covers.
    pub(crate) async fn restrict(
        &self,
        index: usize,
        result: SearchResult,
    ) -> Result<SearchResult> {
        let set = result.row_addrs();
        if set.selected_rows().is_empty() && set.null_rows().is_empty() {
            return Ok(result);
        }
        let allowed = self.live_rows(index).await?;
        let restrict = |set: NullableRowAddrSet| {
            let (mut selected, mut nulls) = set.into_parts();
            selected.mask(&allowed);
            nulls.mask(&allowed);
            NullableRowAddrSet::new(selected, nulls)
        };
        Ok(match result {
            SearchResult::Exact(set) => SearchResult::Exact(restrict(set)),
            SearchResult::AtMost(set) => SearchResult::AtMost(restrict(set)),
            SearchResult::AtLeast(set) => SearchResult::AtLeast(restrict(set)),
        })
    }

    /// Drop lookup matches of segment `index` that are not live rows of its coverage.
    pub(crate) fn restrict_lookup<'a>(
        &'a self,
        index: usize,
        matches: BoxStream<'a, Result<RecordBatch>>,
    ) -> BoxStream<'a, Result<RecordBatch>> {
        let allowed = Arc::new(OnceCell::new());
        matches
            .and_then(move |batch| {
                let allowed = allowed.clone();
                async move {
                    let allowed = allowed.get_or_try_init(|| self.live_rows(index)).await?;
                    let row_ids = batch.column_by_name(ROW_ID).ok_or_else(|| {
                        Error::internal("scalar index lookup batch is missing the _rowid column")
                    })?;
                    let keep = row_ids
                        .as_primitive::<UInt64Type>()
                        .values()
                        .iter()
                        .map(|&row_id| allowed.selected(row_id))
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
    async fn skips_masking_without_a_multi_segment_row_id_scope() {
        let stable = dataset(true).await;
        let single = segment(None);
        assert!(
            SegmentCoverage::resolve(&stable, &[&single], SegmentRowDomain::RowId)
                .await
                .unwrap()
                .is_none()
        );

        let first = segment(None);
        let second = segment(None);
        assert!(
            SegmentCoverage::resolve(&stable, &[&first, &second], SegmentRowDomain::RowAddress)
                .await
                .unwrap()
                .is_none()
        );

        let plain = dataset(false).await;
        assert!(
            SegmentCoverage::resolve(&plain, &[&first, &second], SegmentRowDomain::RowId)
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn refuses_unknown_coverage_when_masking_is_required() {
        let dataset = dataset(true).await;
        let first = segment(None);
        let second = segment(Some(RoaringBitmap::from_iter([0])));
        let error = SegmentCoverage::resolve(&dataset, &[&first, &second], SegmentRowDomain::RowId)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error.to_string().contains("missing fragment coverage"),
            "{error}"
        );

        let first = segment(Some(RoaringBitmap::from_iter([0])));
        let second = segment(Some(RoaringBitmap::from_iter([0])));
        let coverage =
            SegmentCoverage::resolve(&dataset, &[&first, &second], SegmentRowDomain::RowId)
                .await
                .unwrap()
                .unwrap();
        assert_eq!(
            coverage.fragments,
            vec![RoaringBitmap::from_iter([0]), RoaringBitmap::from_iter([0])]
        );
    }
}
