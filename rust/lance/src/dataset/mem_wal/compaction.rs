// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Committing an SSTable prefix into the base table with the row ids it
//! already carries.
//!
//! On a shard that assigns stable row ids ([`super::WAL_ROW_ID`]), every
//! SSTable row was stamped with its id at insert. Compaction therefore has
//! nothing to resolve and nothing to mint, which is what lets it run in a
//! process with no shard claim, no shard manifest and no reservation.
//!
//! `merge_insert` cannot serve this. It preserves the id of a key that already
//! exists in the base table, but *mints* one for a key that does not -- and a
//! key arriving from the WAL is exactly that case. Its id would change the
//! moment it compacted, and a take by the WAL-era id would silently miss. So
//! the merge is replaced by one explicit [`Operation::Update`] carrying:
//!
//! * **new fragments** whose `row_id_meta` already holds their ids, which
//!   `assign_row_ids` then skips rather than re-assigns;
//! * **updated fragments** whose deletion vectors mask every base row the
//!   commit supersedes -- which is also why a tombstone never needs an id of
//!   its own, and why there is no separate delete pass;
//! * the **compacted-SSTable watermark**, on the same commit as the data.
//!
//! The caller owns the address resolution and the retry. An address is
//! version-scoped and a concurrent base commit invalidates it; an id is stable
//! and survives any rewrite by construction. So a conflict means "re-resolve the
//! addresses and rebuild the masks", never "re-assign the ids".

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_array::RecordBatch;
use lance_core::utils::deletion::DeletionVector;
use lance_core::{Error, Result};
use lance_index::mem_wal::CompactedSsTable;
use lance_select::{RowAddrTreeMap, RowSetOps};
use lance_table::format::{Fragment, RowIdMeta};
use lance_table::io::deletion::write_deletion_file;
use lance_table::rowids::{RowIdSequence, write_row_ids};
use roaring::RoaringBitmap;

use crate::dataset::rowids::get_row_id_index;
use crate::dataset::transaction::{Operation, Transaction};
use crate::dataset::write::CommitBuilder;
use crate::dataset::{Dataset, InsertBuilder, WriteMode, WriteParams};
use crate::io::deletion::read_dataset_deletion_file;

/// A compacted SSTable prefix, already deduped newest-per-key by its caller.
pub struct PreAssignedRows {
    /// The surviving rows, projected to the base table's schema.
    pub rows: RecordBatch,
    /// One stable row id per surviving row, in the same order.
    pub row_ids: Vec<u64>,
    /// Addresses of the base rows this commit supersedes: the older copies of
    /// the surviving keys, plus every row a tombstone in the prefix deletes.
    ///
    /// Must be **exact**. Lance intersects it against existing deletions and
    /// unions it into the rebased deletion vectors, so a superset produces
    /// spurious conflicts and a subset lets a concurrent delete remove the
    /// wrong rows.
    pub superseded: RowAddrTreeMap,
    /// The SSTables this commit marks compacted.
    pub compacted_sstables: Vec<CompactedSsTable>,
}

/// Commit `input` against `dataset`, keeping every row's id exactly as given.
///
/// Fails rather than repairs when the ids are not committable: an id repeated
/// within the commit, or one already live in the base table at an address this
/// commit does not supersede, would give one logical row two live copies, and
/// nothing downstream catches that -- `assign_row_ids` skips a fragment whose
/// sequence is already complete.
///
/// Returns a [`CommitConflict`](Error::RetryableCommitConflict) when a
/// concurrent commit moved the base rows out from under `superseded`; the
/// caller re-resolves the addresses and calls again.
pub async fn commit_preassigned_rows(
    dataset: Arc<Dataset>,
    input: PreAssignedRows,
) -> Result<Dataset> {
    let PreAssignedRows {
        rows,
        row_ids,
        superseded,
        compacted_sstables,
    } = input;

    if rows.num_rows() != row_ids.len() {
        return Err(Error::invalid_input(format!(
            "{} rows but {} row ids",
            rows.num_rows(),
            row_ids.len()
        )));
    }
    validate_ids(&dataset, &row_ids, &superseded).await?;

    let new_fragments = write_id_bearing_fragments(&dataset, &rows, &row_ids).await?;
    let updated_fragments = mask_superseded_rows(&dataset, &superseded).await?;

    let operation = Operation::Update {
        removed_fragment_ids: Vec::new(),
        updated_fragments,
        new_fragments,
        fields_modified: Vec::new(),
        compacted_sstables,
        fields_for_preserving_frag_bitmap: Vec::new(),
        update_mode: None,
        inserted_rows_filter: None,
        updated_fragment_offsets: None,
    };
    let transaction = Transaction::new(dataset.manifest.version, operation, None);

    CommitBuilder::new(dataset)
        .with_affected_rows(superseded)
        .execute(transaction)
        .await
}

/// Write the rows as data files and stamp each resulting fragment with the ids
/// its rows already carry.
///
/// [`RowIdMeta::Inline`] rather than `External`: `assign_row_ids` computes an
/// `External` fragment's existing row count as zero and would append a *second*
/// set of ids on top. Nothing writes `External` today, but the skip property
/// this design rests on is specific to `Inline`.
///
/// The ids come out of a hash-ordered dedup, which is the worst case for the
/// sequence's run encoding -- roughly 8 bytes per row, held inline in the
/// manifest. Sorting the rows by id first would recover most of that; it is
/// left undone deliberately, because the sort has to carry every column of a
/// batch that is already materialized, and the manifest growth is linear in
/// compaction size rather than unbounded.
async fn write_id_bearing_fragments(
    dataset: &Arc<Dataset>,
    rows: &RecordBatch,
    row_ids: &[u64],
) -> Result<Vec<Fragment>> {
    if rows.num_rows() == 0 {
        return Ok(Vec::new());
    }

    let schema = rows.schema();
    let reader = arrow_array::RecordBatchIterator::new(vec![Ok(rows.clone())], schema);
    let staged = InsertBuilder::new(dataset.clone())
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        })
        .execute_uncommitted_stream(reader)
        .await?;

    let Operation::Append { mut fragments } = staged.operation else {
        return Err(Error::internal(format!(
            "staging pre-assigned rows produced {} rather than an Append",
            staged.operation.name()
        )));
    };

    // The writer preserves input order across the fragments it produces, so the
    // ids slice in the same order.
    let mut offset = 0usize;
    for fragment in fragments.iter_mut() {
        let count = fragment.physical_rows.ok_or_else(|| {
            Error::internal(format!(
                "fragment {} was written without a physical row count, so its ids \
                 cannot be aligned",
                fragment.id
            ))
        })?;
        let end = offset.checked_add(count).ok_or_else(|| {
            Error::internal("physical row counts overflowed while aligning row ids".to_string())
        })?;
        if end > row_ids.len() {
            return Err(Error::internal(format!(
                "staged fragments hold more rows than the {} ids given for them",
                row_ids.len()
            )));
        }
        let sequence = RowIdSequence::from(&row_ids[offset..end]);
        fragment.row_id_meta = Some(RowIdMeta::Inline(write_row_ids(&sequence).into()));
        offset = end;
    }
    if offset != row_ids.len() {
        return Err(Error::internal(format!(
            "staged fragments hold {offset} rows but {} ids were given",
            row_ids.len()
        )));
    }
    Ok(fragments)
}

/// Write a deletion file per affected base fragment, unioning the superseded
/// offsets into whatever that fragment already masks.
async fn mask_superseded_rows(
    dataset: &Arc<Dataset>,
    superseded: &RowAddrTreeMap,
) -> Result<Vec<Fragment>> {
    let by_id: HashMap<u64, &Fragment> = dataset
        .manifest
        .fragments
        .iter()
        .map(|fragment| (fragment.id, fragment))
        .collect();

    let mut updated = Vec::new();
    for (fragment_id, _) in superseded.iter() {
        let fragment_id = *fragment_id as u64;
        let Some(fragment) = by_id.get(&fragment_id) else {
            // The fragment was rewritten out from under us; the commit below
            // detects it as a conflict and the caller re-resolves.
            continue;
        };
        let Some(offsets) = superseded.get_fragment_bitmap(fragment_id as u32) else {
            continue;
        };

        let mut merged: RoaringBitmap = offsets.clone();
        if let Some(existing) = &fragment.deletion_file {
            let existing = read_dataset_deletion_file(dataset, fragment_id, existing).await?;
            merged |= RoaringBitmap::from(existing.as_ref());
        }

        let mut next = (*fragment).clone();
        next.deletion_file = write_deletion_file(
            &dataset.base,
            fragment_id,
            dataset.manifest.version,
            &DeletionVector::from(merged),
            dataset.object_store.as_ref(),
        )
        .await?;
        updated.push(next);
    }
    Ok(updated)
}

/// The whole invariant: every id is unique within the commit, and is either
/// superseded by it or absent from the base table.
///
/// Deliberately says nothing about any reservation. The reservation is a
/// property of the writing pod and this check has to hold in the job path,
/// where there is none -- and a reservation-relative form would also be
/// *wrong*: a prefix compaction can carry a row whose id was assigned into a
/// generation being compacted in this same pass, so it is neither live in base
/// nor drawn from the current chunk.
async fn validate_ids(
    dataset: &Arc<Dataset>,
    row_ids: &[u64],
    superseded: &RowAddrTreeMap,
) -> Result<()> {
    let mut seen: HashSet<u64> = HashSet::with_capacity(row_ids.len());
    for id in row_ids {
        if !seen.insert(*id) {
            return Err(Error::invalid_input(format!(
                "row id {id} appears twice in one commit"
            )));
        }
    }

    let Some(index) = get_row_id_index(dataset).await? else {
        return Err(Error::invalid_input(
            "committing pre-assigned row ids requires a dataset that uses stable row ids"
                .to_string(),
        ));
    };
    for id in row_ids {
        let Some(address) = index.get(*id)? else {
            continue;
        };
        if !superseded.contains(u64::from(address)) {
            return Err(Error::invalid_input(format!(
                "row id {id} is live in the base table at {address} and this commit does \
                 not supersede it; committing would give one id two live rows"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, RecordBatchIterator};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};

    fn schema() -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Int32, true),
        ]))
    }

    fn batch(rows: &[(i32, i32)]) -> RecordBatch {
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int32Array::from_iter_values(rows.iter().map(|(id, _)| *id))),
                Arc::new(Int32Array::from_iter_values(rows.iter().map(|(_, v)| *v))),
            ],
        )
        .unwrap()
    }

    async fn base(rows: &[(i32, i32)]) -> Arc<Dataset> {
        let initial = batch(rows);
        let reader = RecordBatchIterator::new(vec![Ok(initial)], schema());
        Arc::new(
            Dataset::write(
                reader,
                "memory://",
                Some(WriteParams {
                    enable_stable_row_ids: true,
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        )
    }

    /// Read back every live `(id, value, _rowid)`, sorted by id.
    async fn live_rows(dataset: &Dataset) -> Vec<(i32, i32, u64)> {
        use arrow_array::cast::AsArray;
        use arrow_array::types::{Int32Type, UInt64Type};

        let mut scan = dataset.scan();
        scan.with_row_id();
        let batch = scan.try_into_batch().await.unwrap();
        let ids = batch["id"].as_primitive::<Int32Type>();
        let values = batch["value"].as_primitive::<Int32Type>();
        let row_ids = batch[lance_core::ROW_ID].as_primitive::<UInt64Type>();
        let mut out: Vec<(i32, i32, u64)> = (0..batch.num_rows())
            .map(|i| (ids.value(i), values.value(i), row_ids.value(i)))
            .collect();
        out.sort();
        out
    }

    /// Addresses of the base rows holding the given primary keys.
    async fn addresses_of(dataset: &Dataset, keys: &[i32]) -> RowAddrTreeMap {
        use arrow_array::cast::AsArray;
        use arrow_array::types::{Int32Type, UInt64Type};

        let mut scan = dataset.scan();
        scan.with_row_address();
        let batch = scan.try_into_batch().await.unwrap();
        let ids = batch["id"].as_primitive::<Int32Type>();
        let addrs = batch[lance_core::ROW_ADDR].as_primitive::<UInt64Type>();
        let mut selected = RowAddrTreeMap::new();
        for i in 0..batch.num_rows() {
            if keys.contains(&ids.value(i)) {
                selected.insert(addrs.value(i));
            }
        }
        selected
    }

    /// A key that is new to base keeps the id the WAL assigned it, rather than
    /// being minted a fresh one the way `merge_insert` would.
    #[tokio::test]
    async fn a_new_key_keeps_the_id_it_arrived_with() {
        let dataset = base(&[(1, 10), (2, 20)]).await;
        assert_eq!(dataset.manifest.next_row_id, 2);

        let committed = commit_preassigned_rows(
            dataset,
            PreAssignedRows {
                rows: batch(&[(3, 30), (4, 40)]),
                // Deliberately far above `next_row_id`: a reservation leaves
                // gaps, and the commit must honour the ids rather than
                // re-sequence them.
                row_ids: vec![500, 501],
                superseded: RowAddrTreeMap::new(),
                compacted_sstables: Vec::new(),
            },
        )
        .await
        .unwrap();

        assert_eq!(
            live_rows(&committed).await,
            vec![(1, 10, 0), (2, 20, 1), (3, 30, 500), (4, 40, 501)]
        );
    }

    /// An updated key keeps its id and loses its old copy, in one commit.
    #[tokio::test]
    async fn an_updated_key_keeps_its_id_and_its_old_copy_is_masked() {
        let dataset = base(&[(1, 10), (2, 20)]).await;
        let superseded = addresses_of(&dataset, &[2]).await;

        let committed = commit_preassigned_rows(
            dataset,
            PreAssignedRows {
                rows: batch(&[(2, 99)]),
                row_ids: vec![1],
                superseded,
                compacted_sstables: Vec::new(),
            },
        )
        .await
        .unwrap();

        assert_eq!(
            live_rows(&committed).await,
            vec![(1, 10, 0), (2, 99, 1)],
            "the updated row keeps id 1 and appears exactly once"
        );
    }

    /// A tombstone-winning key is masked with no replacement row, which is why
    /// a tombstone never needs an id and there is no separate delete pass.
    #[tokio::test]
    async fn a_superseded_key_with_no_replacement_row_is_deleted() {
        let dataset = base(&[(1, 10), (2, 20)]).await;
        let superseded = addresses_of(&dataset, &[1]).await;

        let committed = commit_preassigned_rows(
            dataset,
            PreAssignedRows {
                rows: RecordBatch::new_empty(schema()),
                row_ids: Vec::new(),
                superseded,
                compacted_sstables: Vec::new(),
            },
        )
        .await
        .unwrap();

        assert_eq!(live_rows(&committed).await, vec![(2, 20, 1)]);
    }

    /// Two rows sharing an id would give one logical row two live copies, and
    /// nothing downstream catches it -- `assign_row_ids` skips a fragment whose
    /// sequence is already complete.
    #[tokio::test]
    async fn a_repeated_id_within_one_commit_is_rejected() {
        let dataset = base(&[(1, 10)]).await;

        let err = commit_preassigned_rows(
            dataset,
            PreAssignedRows {
                rows: batch(&[(3, 30), (4, 40)]),
                row_ids: vec![500, 500],
                superseded: RowAddrTreeMap::new(),
                compacted_sstables: Vec::new(),
            },
        )
        .await
        .unwrap_err();

        assert!(matches!(err, Error::InvalidInput { .. }), "got {err:?}");
        assert!(err.to_string().contains("appears twice"), "{err}");
    }

    /// Reusing an id that is still live in base without superseding it is the
    /// same corruption from the other direction.
    #[tokio::test]
    async fn an_id_live_in_base_that_is_not_superseded_is_rejected() {
        let dataset = base(&[(1, 10), (2, 20)]).await;

        let err = commit_preassigned_rows(
            dataset,
            PreAssignedRows {
                rows: batch(&[(3, 30)]),
                row_ids: vec![1],
                superseded: RowAddrTreeMap::new(),
                compacted_sstables: Vec::new(),
            },
        )
        .await
        .unwrap_err();

        assert!(matches!(err, Error::InvalidInput { .. }), "got {err:?}");
        assert!(err.to_string().contains("live in the base table"), "{err}");
    }

    /// A dataset without stable row ids has nowhere to put them.
    #[tokio::test]
    async fn a_dataset_without_stable_row_ids_is_rejected() {
        let reader = RecordBatchIterator::new(vec![Ok(batch(&[(1, 10)]))], schema());
        let dataset = Arc::new(Dataset::write(reader, "memory://", None).await.unwrap());

        let err = commit_preassigned_rows(
            dataset,
            PreAssignedRows {
                rows: batch(&[(2, 20)]),
                row_ids: vec![7],
                superseded: RowAddrTreeMap::new(),
                compacted_sstables: Vec::new(),
            },
        )
        .await
        .unwrap_err();

        assert!(matches!(err, Error::InvalidInput { .. }), "got {err:?}");
        assert!(err.to_string().contains("stable row ids"), "{err}");
    }
}
