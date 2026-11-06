// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Resolve requested IDs while preserving complete-index lookup semantics.

use lance_core::utils::address::RowAddress;

use crate::{Dataset, Result};

use super::get_row_id_index_with;

/// Stable-ID writers allocate below `next_row_id`, an exclusive high-water
/// mark preserved through clone/restore. It can reject wholly out-of-range
/// live requests without I/O; it cannot route lower or deleted IDs.
pub async fn resolve_row_ids(
    dataset: &Dataset,
    row_ids: &[u64],
    with_deletions: bool,
) -> Result<Option<Vec<Option<RowAddress>>>> {
    if !dataset.manifest.uses_stable_row_ids() {
        return Ok(None);
    }
    if row_ids.is_empty()
        || (with_deletions && row_ids.iter().all(|id| *id >= dataset.manifest.next_row_id))
    {
        return Ok(Some(vec![None; row_ids.len()]));
    }
    get_row_id_index_with(dataset, with_deletions)
        .await?
        .map(|index| index.get_many(row_ids))
        .transpose()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::cast::AsArray;
    use arrow_array::types::{Int32Type, UInt64Type};
    use lance_table::rowids::RowIdSequence;
    use lance_table::rowids::version::RowDatasetVersionSequence;
    use lance_table::transaction::{Operation, Transaction};

    use super::*;
    use crate::dataset::ProjectionRequest;
    use crate::dataset::rowids::{RowLineage, get_row_id_index, place_row_lineage};
    use crate::dataset::write::CommitBuilder;

    async fn scan_address(dataset: &Dataset, value: i32) -> RowAddress {
        let mut scan = dataset.scan();
        scan.with_row_address()
            .filter(&format!("i = {value}"))
            .unwrap();
        let batch = scan.try_into_batch().await.unwrap();
        assert_eq!(batch.num_rows(), 1);
        RowAddress::from(
            batch[lance_core::ROW_ADDR]
                .as_primitive::<UInt64Type>()
                .value(0),
        )
    }

    #[tokio::test]
    async fn allocation_bound_rejects_only_out_of_range_requests_without_io() {
        let dataset = super::super::cache::tests::dataset(true).await;
        let bound = dataset.manifest.next_row_id;
        let before = dataset.object_store.io_stats_snapshot();
        for ids in [vec![bound, bound], vec![bound, u64::MAX], vec![]] {
            assert_eq!(
                resolve_row_ids(&dataset, &ids, true)
                    .await
                    .unwrap()
                    .unwrap(),
                vec![None; ids.len()]
            );
        }
        assert_eq!(
            dataset.object_store.io_stats_snapshot().read_iops,
            before.read_iops
        );
        let expected = scan_address(&dataset, 7).await;
        assert_eq!(
            resolve_row_ids(&dataset, &[bound, 7, bound], true)
                .await
                .unwrap()
                .unwrap(),
            vec![None, Some(expected), None]
        );
    }

    #[tokio::test]
    async fn deleted_segment_copy_is_filtered_before_duplicate_validation() {
        let mut dataset = super::super::cache::tests::dataset(false).await;
        let expected = scan_address(&dataset, 7).await;
        dataset.delete("i < 5").await.unwrap();
        get_row_id_index(&dataset).await.unwrap().unwrap();
        // Avoid compaction: its fragment order need not match payload order.
        // Keep two explicit inline segments in the original ten-row fragment.
        let mut fragments = dataset.manifest.fragments.as_ref().clone();
        assert_eq!(fragments[0].physical_rows, Some(10));
        assert_eq!(
            expected,
            RowAddress::new_from_parts(fragments[0].id as u32, 7)
        );
        assert!(fragments[0].deletion_file.is_some());
        let mut sequence = RowIdSequence::from(0..5);
        sequence.extend(RowIdSequence::from(0..5));
        let lineage = RowLineage {
            row_ids: sequence,
            created_at: RowDatasetVersionSequence::from_uniform_row_count(10, 1),
            last_updated_at: RowDatasetVersionSequence::from_uniform_row_count(10, 1),
        };
        place_row_lineage(&dataset, &lineage)
            .await
            .unwrap()
            .apply(&mut fragments[0]);
        let transaction = Transaction::new_from_version(
            dataset.version().version,
            Operation::Merge {
                fragments,
                schema: dataset.schema().clone(),
                preserves_nullability: true,
            },
        );
        dataset = CommitBuilder::new(Arc::new(dataset))
            .execute(transaction)
            .await
            .unwrap();
        let batch = dataset
            .take_rows(
                &[2, 2],
                ProjectionRequest::from_columns(["i"], dataset.schema()),
            )
            .await
            .unwrap();
        assert_eq!(
            batch["i"].as_primitive::<Int32Type>().values().as_ref(),
            &[7, 7]
        );
    }
}
