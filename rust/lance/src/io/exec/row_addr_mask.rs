// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! RowAddrMask prefilter wiring for vector search.
//!
//! An externally supplied [`RowAddrMask`] is applied to a KNN search through two
//! pieces, because the two search branches consume a prefilter differently:
//!   - [`MaskAndLoader`] folds the mask into the index-side prefilter loader
//!     (ANN / IVF branch). The mask, any filter-derived selection vector, and
//!     the deletion vector are all combined (logical AND) by DatasetPreFilter.
//!   - [`RowAddrMaskFilterExec`] applies the mask to the flat-KNN branch, which
//!     scans fragments not covered by the vector index and so never reaches the
//!     index-side prefilter.

use std::sync::Arc;

use arrow::datatypes::UInt64Type;
use arrow_array::cast::AsArray;
use arrow_array::{BooleanArray, RecordBatch};
use async_trait::async_trait;
use datafusion::error::DataFusionError;
use datafusion::execution::TaskContext;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
};
use futures::StreamExt;
use lance_core::error::DataFusionResult;
use lance_core::{ROW_ID, Result};
use lance_index::prefilter::FilterLoader;
use lance_select::RowAddrMask;

/// FilterLoader that combines an external RowAddrMask (logical AND) with an
/// optional inner loader.
///
/// With an inner loader present the two masks are intersected; otherwise the
/// external mask is used alone. DatasetPreFilter later intersects the result
/// with the dataset deletion vector.
pub struct MaskAndLoader {
    mask: Arc<RowAddrMask>,
    inner: Option<Box<dyn FilterLoader>>,
}

impl MaskAndLoader {
    pub fn new(mask: Arc<RowAddrMask>, inner: Option<Box<dyn FilterLoader>>) -> Self {
        Self { mask, inner }
    }
}

#[async_trait]
impl FilterLoader for MaskAndLoader {
    async fn load(self: Box<Self>) -> Result<RowAddrMask> {
        match self.inner {
            Some(inner) => Ok(Arc::unwrap_or_clone(self.mask) & inner.load().await?),
            None => Ok(Arc::unwrap_or_clone(self.mask)),
        }
    }
}

/// Execution node that drops rows whose `_rowid` is not selected by `mask`.
///
/// The key is read from the `_rowid` column, and `mask` is keyed in that same
/// `_rowid` space, so this is consistent whether stable row ids are enabled (the
/// value is the stable row id) or disabled (it is the row address). Schema and
/// ordering are preserved; only the row count changes.
#[derive(Debug)]
pub struct RowAddrMaskFilterExec {
    input: Arc<dyn ExecutionPlan>,
    mask: Arc<RowAddrMask>,
    properties: Arc<PlanProperties>,
}

impl RowAddrMaskFilterExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, mask: Arc<RowAddrMask>) -> Self {
        // Filtering preserves schema, partitioning and ordering, so the input's
        // plan properties carry over unchanged.
        let properties = input.properties().clone();
        Self {
            input,
            mask,
            properties,
        }
    }
}

impl DisplayAs for RowAddrMaskFilterExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "RowAddrMaskFilter")
    }
}

impl ExecutionPlan for RowAddrMaskFilterExec {
    fn name(&self) -> &str {
        "RowAddrMaskFilterExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(
                "RowAddrMaskFilterExec must have exactly one child".to_string(),
            ));
        }
        let child = children.pop().ok_or_else(|| {
            DataFusionError::Internal("RowAddrMaskFilterExec child unavailable".to_string())
        })?;
        Ok(Arc::new(Self::new(child, self.mask.clone())))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let input_stream = self.input.execute(partition, context)?;
        let schema = input_stream.schema();
        let mask = self.mask.clone();
        let stream = input_stream.map(move |batch| apply_mask(&mask, batch?));
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }
}

/// Keep rows whose `_rowid` is selected by the mask (the mask is keyed in the
/// same `_rowid` space). Null ids are dropped; they cannot be in any allow set.
fn apply_mask(mask: &RowAddrMask, batch: RecordBatch) -> DataFusionResult<RecordBatch> {
    let row_id_column = batch.column_by_name(ROW_ID).ok_or_else(|| {
        DataFusionError::Internal(format!(
            "RowAddrMaskFilterExec input missing {ROW_ID} column"
        ))
    })?;
    let row_ids = row_id_column
        .as_primitive_opt::<UInt64Type>()
        .ok_or_else(|| {
            DataFusionError::Internal(format!(
                "{ROW_ID} column must be UInt64 but was {:?}",
                row_id_column.data_type()
            ))
        })?;
    let keep = BooleanArray::from_iter(
        row_ids
            .iter()
            .map(|addr| Some(addr.is_some_and(|addr| mask.selected(addr)))),
    );
    arrow::compute::filter_record_batch(&batch, &keep)
        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow_array::{Int32Array, UInt64Array};
    use lance_select::RowAddrTreeMap;

    fn batch_with_rowids(ids: Vec<Option<u64>>) -> RecordBatch {
        let n = ids.len() as i32;
        let schema = Arc::new(Schema::new(vec![
            Field::new(ROW_ID, DataType::UInt64, true),
            Field::new("v", DataType::Int32, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt64Array::from(ids)),
                Arc::new(Int32Array::from((0..n).collect::<Vec<_>>())),
            ],
        )
        .unwrap()
    }

    fn kept_rowids(batch: &RecordBatch) -> Vec<Option<u64>> {
        batch
            .column_by_name(ROW_ID)
            .unwrap()
            .as_primitive::<UInt64Type>()
            .iter()
            .collect()
    }

    #[test]
    fn apply_mask_allow_keeps_only_selected() {
        let mask = RowAddrMask::from_allowed(RowAddrTreeMap::from_iter([1u64, 3, 5]));
        let batch = batch_with_rowids(vec![Some(1), Some(2), Some(3), Some(4), Some(5)]);
        let out = apply_mask(&mask, batch).unwrap();
        assert_eq!(kept_rowids(&out), vec![Some(1), Some(3), Some(5)]);
    }

    #[test]
    fn apply_mask_block_drops_selected() {
        let mask = RowAddrMask::from_block(RowAddrTreeMap::from_iter([2u64, 4]));
        let batch = batch_with_rowids(vec![Some(1), Some(2), Some(3), Some(4), Some(5)]);
        let out = apply_mask(&mask, batch).unwrap();
        assert_eq!(kept_rowids(&out), vec![Some(1), Some(3), Some(5)]);
    }

    #[test]
    fn apply_mask_drops_null_rowids() {
        // A null id cannot be in any allow set, so it is dropped.
        let mask = RowAddrMask::from_allowed(RowAddrTreeMap::from_iter([1u64, 2, 3]));
        let batch = batch_with_rowids(vec![Some(1), None, Some(3)]);
        let out = apply_mask(&mask, batch).unwrap();
        assert_eq!(kept_rowids(&out), vec![Some(1), Some(3)]);
    }

    #[test]
    fn apply_mask_missing_rowid_column_errs() {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1, 2]))]).unwrap();
        let mask = RowAddrMask::from_allowed(RowAddrTreeMap::from_iter([1u64]));
        let err = apply_mask(&mask, batch).unwrap_err();
        assert!(matches!(err, DataFusionError::Internal(_)), "got {err:?}");
        let msg = err.to_string();
        assert!(
            msg.contains(ROW_ID) && msg.contains("missing"),
            "unexpected: {msg}"
        );
    }

    #[test]
    fn apply_mask_wrong_type_rowid_column_errs() {
        // _rowid present but not UInt64 -> Internal error naming the actual type.
        let schema = Arc::new(Schema::new(vec![Field::new(
            ROW_ID,
            DataType::Int32,
            false,
        )]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1, 2]))]).unwrap();
        let mask = RowAddrMask::from_allowed(RowAddrTreeMap::from_iter([1u64]));
        let err = apply_mask(&mask, batch).unwrap_err();
        assert!(matches!(err, DataFusionError::Internal(_)), "got {err:?}");
        let msg = err.to_string();
        assert!(
            msg.contains("UInt64") && msg.contains("Int32"),
            "unexpected: {msg}"
        );
    }

    struct FixedLoader(RowAddrMask);

    #[async_trait]
    impl FilterLoader for FixedLoader {
        async fn load(self: Box<Self>) -> Result<RowAddrMask> {
            Ok(self.0)
        }
    }

    #[tokio::test]
    async fn mask_and_loader_without_inner_returns_mask() {
        let mask = RowAddrMask::from_allowed(RowAddrTreeMap::from_iter([1u64, 2, 3]));
        let loaded = Box::new(MaskAndLoader::new(Arc::new(mask), None))
            .load()
            .await
            .unwrap();
        assert!(loaded.selected(2));
        assert!(!loaded.selected(4));
    }

    #[tokio::test]
    async fn mask_and_loader_with_inner_intersects() {
        // {1,2,3,4} AND inner {2,4,6} = {2,4}.
        let mask = RowAddrMask::from_allowed(RowAddrTreeMap::from_iter([1u64, 2, 3, 4]));
        let inner = RowAddrMask::from_allowed(RowAddrTreeMap::from_iter([2u64, 4, 6]));
        let loaded = Box::new(MaskAndLoader::new(
            Arc::new(mask),
            Some(Box::new(FixedLoader(inner))),
        ))
        .load()
        .await
        .unwrap();
        assert!(loaded.selected(2));
        assert!(loaded.selected(4));
        assert!(!loaded.selected(1));
        assert!(!loaded.selected(6));
    }

    #[tokio::test]
    async fn mask_and_loader_block_and_allow() {
        // block{2} AND allow{1,2,3} = allow({1,2,3} - {2}) = allow{1,3}.
        let mask = RowAddrMask::from_block(RowAddrTreeMap::from_iter([2u64]));
        let inner = RowAddrMask::from_allowed(RowAddrTreeMap::from_iter([1u64, 2, 3]));
        let loaded = Box::new(MaskAndLoader::new(
            Arc::new(mask),
            Some(Box::new(FixedLoader(inner))),
        ))
        .load()
        .await
        .unwrap();
        assert!(loaded.selected(1));
        assert!(loaded.selected(3));
        assert!(!loaded.selected(2));
        assert!(!loaded.selected(4));
    }

    #[tokio::test]
    async fn mask_and_loader_block_and_block() {
        // block{1} AND block{2} = block{1,2}: everything except 1 and 2 is selected.
        let mask = RowAddrMask::from_block(RowAddrTreeMap::from_iter([1u64]));
        let inner = RowAddrMask::from_block(RowAddrTreeMap::from_iter([2u64]));
        let loaded = Box::new(MaskAndLoader::new(
            Arc::new(mask),
            Some(Box::new(FixedLoader(inner))),
        ))
        .load()
        .await
        .unwrap();
        assert!(!loaded.selected(1));
        assert!(!loaded.selected(2));
        assert!(loaded.selected(3));
    }
}
